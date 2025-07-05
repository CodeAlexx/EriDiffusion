//! Plugin system for extensibility

use crate::{Result, Error};
use async_trait::async_trait;
use dashmap::DashMap;
use libloading::Library;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// Global plugin registry
static PLUGIN_REGISTRY: Lazy<PluginRegistry> = Lazy::new(PluginRegistry::new);

/// Plugin trait that all extensions must implement
#[async_trait]
pub trait Plugin: Send + Sync + Any {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;
    
    /// Initialize the plugin
    async fn initialize(&mut self, context: &PluginContext) -> Result<()>;
    
    /// Shutdown the plugin
    async fn shutdown(&mut self) -> Result<()>;
    
    /// Get plugin capabilities
    fn capabilities(&self) -> PluginCapabilities;
    
    /// Handle events
    async fn on_event(&mut self, event: PluginEvent) -> Result<()> {
        Ok(())
    }
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub license: String,
    pub homepage: Option<String>,
    pub dependencies: Vec<PluginDependency>,
}

/// Plugin dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    pub name: String,
    pub version: String,
    pub optional: bool,
}

/// Plugin capabilities
#[derive(Debug, Clone, Default)]
pub struct PluginCapabilities {
    pub model_support: Vec<crate::ModelArchitecture>,
    pub network_support: Vec<crate::NetworkType>,
    pub hooks: Vec<HookType>,
    pub custom_losses: bool,
    pub custom_optimizers: bool,
    pub custom_schedulers: bool,
}

/// Hook types that plugins can register for
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookType {
    PreTraining,
    PostTrainingStep,
    PreValidation,
    PostValidation,
    PreGeneration,
    PostGeneration,
    ModelLoaded,
    ModelSaved,
    LossComputation,
    OptimizerStep,
}

/// Plugin events
#[derive(Debug, Clone)]
pub enum PluginEvent {
    TrainingStarted { 
        model: String,
        config: HashMap<String, serde_json::Value>,
    },
    TrainingStep {
        step: usize,
        loss: f32,
        metrics: HashMap<String, f32>,
    },
    TrainingCompleted {
        total_steps: usize,
        final_loss: f32,
    },
    GenerationStarted {
        prompt: String,
        config: HashMap<String, serde_json::Value>,
    },
    GenerationCompleted {
        images: usize,
        time_ms: u64,
    },
    Custom(String, serde_json::Value),
}

/// Plugin context provided during initialization
pub struct PluginContext {
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub config: HashMap<String, serde_json::Value>,
    pub device_manager: Arc<crate::DeviceManager>,
}

/// Plugin registry for managing loaded plugins
pub struct PluginRegistry {
    plugins: DashMap<Uuid, Arc<RwLock<Box<dyn Plugin>>>>,
    libraries: DashMap<Uuid, Library>,
    hooks: DashMap<HookType, Vec<Uuid>>,
    search_paths: RwLock<Vec<PathBuf>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: DashMap::new(),
            libraries: DashMap::new(),
            hooks: DashMap::new(),
            search_paths: RwLock::new(vec![]),
        }
    }
    
    /// Get the global plugin registry
    pub fn global() -> &'static Self {
        &PLUGIN_REGISTRY
    }
    
    /// Add a search path for plugins
    pub fn add_search_path(&self, path: PathBuf) {
        self.search_paths.write().push(path);
    }
    
    /// Load a plugin from a dynamic library
    pub async fn load_plugin(&self, path: &Path) -> Result<Uuid> {
        // Verify the plugin file exists
        if !path.exists() {
            return Err(Error::Plugin(format!("Plugin file not found: {:?}", path)));
        }
        
        // Load the library
        let library = unsafe {
            Library::new(path).map_err(|e| Error::Plugin(format!("Failed to load plugin: {}", e)))?
        };
        
        // Get the plugin creation function
        let create_plugin: libloading::Symbol<fn() -> Box<dyn Plugin>> = unsafe {
            library.get(b"create_plugin")
                .map_err(|e| Error::Plugin(format!("Plugin missing create_plugin function: {}", e)))?
        };
        
        // Create the plugin instance
        let mut plugin = create_plugin();
        let metadata = plugin.metadata();
        let plugin_id = metadata.id;
        
        // Initialize the plugin
        let context = PluginContext {
            data_dir: dirs::data_dir().unwrap_or_default().join("eridiffusion").join("plugins").join(&metadata.name),
            cache_dir: dirs::cache_dir().unwrap_or_default().join("eridiffusion").join("plugins").join(&metadata.name),
            config: HashMap::new(),
            device_manager: Arc::new(crate::DeviceManager::new()?),
        };
        
        plugin.initialize(&context).await?;
        
        // Register hooks
        let capabilities = plugin.capabilities();
        for hook in capabilities.hooks {
            self.hooks.entry(hook).or_default().push(plugin_id);
        }
        
        // Store the plugin
        self.plugins.insert(plugin_id, Arc::new(RwLock::new(plugin)));
        self.libraries.insert(plugin_id, library);
        
        tracing::info!("Loaded plugin: {} v{}", metadata.name, metadata.version);
        
        Ok(plugin_id)
    }
    
    /// Unload a plugin
    pub async fn unload_plugin(&self, id: Uuid) -> Result<()> {
        // Get the plugin
        let plugin = self.plugins.remove(&id)
            .ok_or_else(|| Error::Plugin(format!("Plugin not found: {}", id)))?;
        
        // Shutdown the plugin
        plugin.1.write().shutdown().await?;
        
        // Remove from hooks
        for mut hooks in self.hooks.iter_mut() {
            hooks.retain(|pid| *pid != id);
        }
        
        // Remove the library
        self.libraries.remove(&id);
        
        Ok(())
    }
    
    /// Get a plugin by ID
    pub fn get_plugin(&self, id: Uuid) -> Option<Arc<RwLock<Box<dyn Plugin>>>> {
        self.plugins.get(&id).map(|p| p.clone())
    }
    
    /// Get all plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        self.plugins.iter()
            .map(|p| p.read().metadata())
            .collect()
    }
    
    /// Trigger a hook
    pub async fn trigger_hook(&self, hook: HookType, event: PluginEvent) -> Result<()> {
        if let Some(plugin_ids) = self.hooks.get(&hook) {
            for plugin_id in plugin_ids.iter() {
                if let Some(plugin) = self.plugins.get(plugin_id) {
                    plugin.write().on_event(event.clone()).await?;
                }
            }
        }
        Ok(())
    }
    
    /// Find plugins by capability
    pub fn find_plugins_by_capability(&self, capability: impl Fn(&PluginCapabilities) -> bool) -> Vec<Uuid> {
        self.plugins.iter()
            .filter(|p| capability(&p.read().capabilities()))
            .map(|p| p.key().clone())
            .collect()
    }
}

/// Initialize the plugin system
pub fn initialize_plugin_system() -> Result<()> {
    // Add default search paths
    let registry = PluginRegistry::global();
    
    // System plugin directory
    if let Some(data_dir) = dirs::data_dir() {
        registry.add_search_path(data_dir.join("eridiffusion").join("plugins"));
    }
    
    // User plugin directory
    if let Some(home_dir) = dirs::home_dir() {
        registry.add_search_path(home_dir.join(".eridiffusion").join("plugins"));
    }
    
    // Current directory plugins
    registry.add_search_path(PathBuf::from("./plugins"));
    
    Ok(())
}

/// Macro to simplify plugin creation
#[macro_export]
macro_rules! declare_plugin {
    ($plugin_type:ty) => {
        #[no_mangle]
        pub extern "C" fn create_plugin() -> Box<dyn $crate::Plugin> {
            Box::new(<$plugin_type>::new())
        }
    };
}