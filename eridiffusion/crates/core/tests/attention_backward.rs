use eridiffusion_core::autograd_ops::attention_backward_recompute;
use eridiffusion_core::device::shared_cuda_device;
use flame_core::{Shape, Tensor};

fn make_tensor_from_fn(shape: &[usize], f: impl Fn(usize) -> f32) -> Tensor {
    let dev = shared_cuda_device().expect("cuda device");
    let numel: usize = shape.iter().product();
    let mut v = Vec::with_capacity(numel);
    for i in 0..numel { v.push(f(i)); }
    Tensor::from_vec(v, Shape::from_dims(shape), dev).expect("tensor")
}

fn sdpa_forward(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
    let kt = k.transpose_dims(2, 3).expect("kt");
    let logits = q.bmm(&kt).expect("q@kt").mul_scalar(scale).expect("scale");
    let attn = logits.softmax(-1).expect("softmax");
    attn.bmm(v).expect("attn@v")
}

fn sum_o_dot_do(o: &Tensor, do_: &Tensor) -> f32 {
    let prod = o.mul(do_).expect("mul");
    prod.sum_all().expect("sum").to_scalar::<f32>().expect("item")
}

#[test]
fn test_attention_backward_fd_check() {
    // Shapes: [B,H,Sq,D] = [1,2,3,4]
    let b = 1usize; let h = 2usize; let sq = 3usize; let d = 4usize; let sk = 3usize;
    let shape_q = [b, h, sq, d];
    let shape_kv = [b, h, sk, d];
    let shape_do = [b, h, sq, d];
    let scale = 1.0_f32 / (d as f32).sqrt();

    // Deterministic fillers
    let q = make_tensor_from_fn(&shape_q, |i| ((i % 7) as f32) * 0.1 - 0.3);
    let k = make_tensor_from_fn(&shape_kv, |i| ((i % 5) as f32) * 0.2 - 0.4);
    let v = make_tensor_from_fn(&shape_kv, |i| ((i % 11) as f32) * 0.05 - 0.25);
    let do_ = make_tensor_from_fn(&shape_do, |i| ((i % 13) as f32) * 0.03 - 0.18);

    // Analytic grads
    let (dq, dk, dv) = attention_backward_recompute(&q, &k, &v, &do_, None, scale).expect("backward");

    // Finite difference params
    let eps = 1e-3_f32;
    let tol = 2e-2_f32;

    // Helper to rebuild tensor with one element perturbed
    let dev = shared_cuda_device().expect("cuda device");

    // Check one element in Q at [0,0,0,0]
    let idx_q = 0usize; // linear idx for the first element
    {
        let mut qv = q.to_vec().expect("q vec");
        let base = qv[idx_q];
        // plus
        qv[idx_q] = base + eps;
        let q_plus = Tensor::from_vec(qv.clone(), q.shape().clone(), dev.clone()).expect("q+");
        let o_plus = sdpa_forward(&q_plus, &k, &v, scale);
        let l_plus = sum_o_dot_do(&o_plus, &do_);
        // minus
        let mut qv2 = q.to_vec().expect("q vec2");
        qv2[idx_q] = base - eps;
        let q_minus = Tensor::from_vec(qv2, q.shape().clone(), dev.clone()).expect("q-");
        let o_minus = sdpa_forward(&q_minus, &k, &v, scale);
        let l_minus = sum_o_dot_do(&o_minus, &do_);
        let d_fd = (l_plus - l_minus) / (2.0 * eps);

        let dqv = dq.to_vec().expect("dq vec");
        let d_an = dqv[idx_q];
        assert!( (d_an - d_fd).abs() < tol, "dQ mismatch: analytic={} fd={}", d_an, d_fd );
    }

    // Check one element in K at [0,1,2,3] -> compute linear idx
    {
        let b0=0usize; let h1=1usize; let s2=2usize; let d3=3usize;
        let idx_k = (((b0 * h + h1) * sk + s2) * d + d3) as usize;
        let mut kv = k.to_vec().expect("k vec");
        let base = kv[idx_k];
        // plus
        kv[idx_k] = base + eps;
        let k_plus = Tensor::from_vec(kv.clone(), k.shape().clone(), dev.clone()).expect("k+");
        let o_plus = sdpa_forward(&q, &k_plus, &v, scale);
        let l_plus = sum_o_dot_do(&o_plus, &do_);
        // minus
        let mut kv2 = k.to_vec().expect("k vec2");
        kv2[idx_k] = base - eps;
        let k_minus = Tensor::from_vec(kv2, k.shape().clone(), dev.clone()).expect("k-");
        let o_minus = sdpa_forward(&q, &k_minus, &v, scale);
        let l_minus = sum_o_dot_do(&o_minus, &do_);
        let d_fd = (l_plus - l_minus) / (2.0 * eps);

        let dkv = dk.to_vec().expect("dk vec");
        let d_an = dkv[idx_k];
        assert!( (d_an - d_fd).abs() < tol, "dK mismatch: analytic={} fd={}", d_an, d_fd );
    }

    // Check one element in V at [0,0,1,2]
    {
        let b0=0usize; let h0=0usize; let s1=1usize; let d2=2usize;
        let idx_v = (((b0 * h + h0) * sk + s1) * d + d2) as usize;
        let mut vv = v.to_vec().expect("v vec");
        let base = vv[idx_v];
        // plus
        vv[idx_v] = base + eps;
        let v_plus = Tensor::from_vec(vv.clone(), v.shape().clone(), dev.clone()).expect("v+");
        let o_plus = sdpa_forward(&q, &k, &v_plus, scale);
        let l_plus = sum_o_dot_do(&o_plus, &do_);
        // minus
        let mut vv2 = v.to_vec().expect("v vec2");
        vv2[idx_v] = base - eps;
        let v_minus = Tensor::from_vec(vv2, v.shape().clone(), dev.clone()).expect("v-");
        let o_minus = sdpa_forward(&q, &k, &v_minus, scale);
        let l_minus = sum_o_dot_do(&o_minus, &do_);
        let d_fd = (l_plus - l_minus) / (2.0 * eps);

        let dvv = dv.to_vec().expect("dv vec");
        let d_an = dvv[idx_v];
        assert!( (d_an - d_fd).abs() < tol, "dV mismatch: analytic={} fd={}", d_an, d_fd );
    }
}
#![cfg(all())]
