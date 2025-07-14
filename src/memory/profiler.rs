//! Memory profiler for debugging and optimization

use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::ffi::c_void;
use serde::{Serialize, Deserialize};

/// Memory event for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub timestamp: Duration,
    pub event_type: MemoryEventType,
    pub size: usize,
    pub device_id: i32,
    pub ptr: usize,
}

/// Type of memory event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryEventType {
    Allocate,
    Deallocate,
    CacheHit,
    CacheMiss,
    GarbageCollect,
}

/// Memory profiler for tracking allocations
pub struct MemoryProfiler {
    events: Mutex<Vec<MemoryEvent>>,
    start_time: Instant,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            start_time: Instant::now(),
        }
    }

    pub fn record_event(&self, event_type: MemoryEventType, size: usize, device_id: i32, ptr: *mut c_void) {
        let event = MemoryEvent {
            timestamp: self.start_time.elapsed(),
            event_type,
            size,
            device_id,
            ptr: ptr as usize,
        };

        self.events.lock().unwrap().push(event);
    }

    pub fn get_events(&self) -> Vec<MemoryEvent> {
        self.events.lock().unwrap().clone()
    }

    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    pub fn export_to_json(&self) -> Result<String, serde_json::Error> {
        let events = self.get_events();
        serde_json::to_string_pretty(&events)
    }
}