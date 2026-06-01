use crate::shaders::{GRID_SHADER, OBJECT_SHADER};
use crate::JsResult;
use js_sys::{Array, Float32Array, Object, Uint32Array};
use wasm_bindgen::prelude::*;

pub const GPU_BUFFER_USAGE_COPY_DST: u32 = 8;
pub const GPU_BUFFER_USAGE_INDEX: u32 = 16;
pub const GPU_BUFFER_USAGE_VERTEX: u32 = 32;
pub const GPU_BUFFER_USAGE_UNIFORM: u32 = 64;
pub const GPU_SHADER_STAGE_VERTEX: u32 = 1;
pub const GPU_SHADER_STAGE_FRAGMENT: u32 = 2;
pub const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT: u32 = 16;

pub fn create_pipeline(
    device: &JsValue,
    color_format: &str,
    depth_format: &str,
    frame_layout: &JsValue,
    grid_layout: &JsValue,
) -> JsResult<JsValue> {
    let module_desc = object();
    set(&module_desc, "label", "Pristine Grid")?;
    set(&module_desc, "code", GRID_SHADER)?;
    let module =
        crate::js_bridge::call_method(device, "createShaderModule", &[module_desc.into()])?;

    let layouts = Array::new();
    layouts.push(frame_layout);
    layouts.push(grid_layout);
    let pipeline_layout_desc = object();
    set(&pipeline_layout_desc, "bindGroupLayouts", &layouts)?;
    let pipeline_layout = crate::js_bridge::call_method(
        device,
        "createPipelineLayout",
        &[pipeline_layout_desc.into()],
    )?;

    let pos_attr = object();
    set(&pos_attr, "shaderLocation", 0)?;
    set(&pos_attr, "offset", 0)?;
    set(&pos_attr, "format", "float32x3")?;
    let uv_attr = object();
    set(&uv_attr, "shaderLocation", 1)?;
    set(&uv_attr, "offset", 12)?;
    set(&uv_attr, "format", "float32x2")?;
    let attrs = Array::new();
    attrs.push(&pos_attr);
    attrs.push(&uv_attr);
    let vertex_buffer = object();
    set(&vertex_buffer, "arrayStride", 20)?;
    set(&vertex_buffer, "attributes", &attrs)?;
    let vertex_buffers = Array::new();
    vertex_buffers.push(&vertex_buffer);
    let vertex = object();
    set(&vertex, "module", &module)?;
    set(&vertex, "entryPoint", "vertexMain")?;
    set(&vertex, "buffers", &vertex_buffers)?;

    let color_target = object();
    set(&color_target, "format", format!("{color_format}-srgb"))?;
    let targets = Array::new();
    targets.push(&color_target);
    let fragment = object();
    set(&fragment, "module", &module)?;
    set(&fragment, "entryPoint", "fragmentMain")?;
    set(&fragment, "targets", &targets)?;

    let depth = object();
    set(&depth, "format", depth_format)?;
    set(&depth, "depthWriteEnabled", true)?;
    set(&depth, "depthCompare", "less-equal")?;

    let desc = object();
    set(&desc, "label", "Pristine Grid")?;
    set(&desc, "layout", &pipeline_layout)?;
    set(&desc, "vertex", &vertex)?;
    set(&desc, "fragment", &fragment)?;
    set(&desc, "depthStencil", &depth)?;
    crate::js_bridge::call_method(device, "createRenderPipeline", &[desc.into()])
}

pub fn create_object_pipeline(
    device: &JsValue,
    color_format: &str,
    depth_format: &str,
    frame_layout: &JsValue,
) -> JsResult<JsValue> {
    let module_desc = object();
    set(&module_desc, "label", "Arcade 3D Objects")?;
    set(&module_desc, "code", OBJECT_SHADER)?;
    let module =
        crate::js_bridge::call_method(device, "createShaderModule", &[module_desc.into()])?;

    let layouts = Array::new();
    layouts.push(frame_layout);
    let pipeline_layout_desc = object();
    set(&pipeline_layout_desc, "bindGroupLayouts", &layouts)?;
    let pipeline_layout = crate::js_bridge::call_method(
        device,
        "createPipelineLayout",
        &[pipeline_layout_desc.into()],
    )?;

    let pos_attr = object();
    set(&pos_attr, "shaderLocation", 0)?;
    set(&pos_attr, "offset", 0)?;
    set(&pos_attr, "format", "float32x3")?;
    let color_attr = object();
    set(&color_attr, "shaderLocation", 1)?;
    set(&color_attr, "offset", 12)?;
    set(&color_attr, "format", "float32x4")?;
    let attrs = Array::new();
    attrs.push(&pos_attr);
    attrs.push(&color_attr);
    let vertex_buffer = object();
    set(&vertex_buffer, "arrayStride", 28)?;
    set(&vertex_buffer, "attributes", &attrs)?;
    let vertex_buffers = Array::new();
    vertex_buffers.push(&vertex_buffer);

    let vertex = object();
    set(&vertex, "module", &module)?;
    set(&vertex, "entryPoint", "vertexMain")?;
    set(&vertex, "buffers", &vertex_buffers)?;

    let color_target = object();
    set(&color_target, "format", format!("{color_format}-srgb"))?;
    let targets = Array::new();
    targets.push(&color_target);
    let fragment = object();
    set(&fragment, "module", &module)?;
    set(&fragment, "entryPoint", "fragmentMain")?;
    set(&fragment, "targets", &targets)?;

    let depth = object();
    set(&depth, "format", depth_format)?;
    set(&depth, "depthWriteEnabled", true)?;
    set(&depth, "depthCompare", "less-equal")?;

    let desc = object();
    set(&desc, "label", "Arcade 3D Objects")?;
    set(&desc, "layout", &pipeline_layout)?;
    set(&desc, "vertex", &vertex)?;
    set(&desc, "fragment", &fragment)?;
    set(&desc, "depthStencil", &depth)?;
    crate::js_bridge::call_method(device, "createRenderPipeline", &[desc.into()])
}

pub fn create_bind_group_layout(
    device: &JsValue,
    label: &str,
    visibility: u32,
) -> JsResult<JsValue> {
    let buffer = object();
    let entry = object();
    set(&entry, "binding", 0)?;
    set(&entry, "visibility", visibility)?;
    set(&entry, "buffer", &buffer)?;
    let entries = Array::new();
    entries.push(&entry);
    let desc = object();
    set(&desc, "label", label)?;
    set(&desc, "entries", &entries)?;
    crate::js_bridge::call_method(device, "createBindGroupLayout", &[desc.into()])
}

pub fn create_bind_group(
    device: &JsValue,
    label: &str,
    layout: &JsValue,
    buffer: &JsValue,
) -> JsResult<JsValue> {
    let resource = object();
    set(&resource, "buffer", buffer)?;
    let entry = object();
    set(&entry, "binding", 0)?;
    set(&entry, "resource", &resource)?;
    let entries = Array::new();
    entries.push(&entry);
    let desc = object();
    set(&desc, "label", label)?;
    set(&desc, "layout", layout)?;
    set(&desc, "entries", &entries)?;
    crate::js_bridge::call_method(device, "createBindGroup", &[desc.into()])
}

pub fn create_buffer(device: &JsValue, label: &str, size: u32, usage: u32) -> JsResult<JsValue> {
    let desc = object();
    set(&desc, "label", label)?;
    set(&desc, "size", size)?;
    set(&desc, "usage", usage)?;
    crate::js_bridge::call_method(device, "createBuffer", &[desc.into()])
}

pub fn write_f32_buffer(queue: &JsValue, buffer: &JsValue, data: &[f32]) -> JsResult<()> {
    let array = Float32Array::from(data);
    crate::js_bridge::call_method(
        queue,
        "writeBuffer",
        &[buffer.clone(), 0.into(), array.into()],
    )?;
    Ok(())
}

pub fn write_u32_buffer(queue: &JsValue, buffer: &JsValue, data: &[u32]) -> JsResult<()> {
    let array = Uint32Array::from(data);
    crate::js_bridge::call_method(
        queue,
        "writeBuffer",
        &[buffer.clone(), 0.into(), array.into()],
    )?;
    Ok(())
}

pub fn perspective_zo(fovy: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fovy * 0.5).tan();
    [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        (far * near) / (near - far),
        0.0,
    ]
}

pub fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize3([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);
    [
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        -dot3(s, eye),
        -dot3(u, eye),
        dot3(f, eye),
        1.0,
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt().max(0.0001);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn object() -> Object {
    Object::new()
}

fn set<T: Into<JsValue>>(object: &Object, key: &str, value: T) -> JsResult<()> {
    js_sys::Reflect::set(object, &JsValue::from_str(key), &value.into()).map(|_| ())
}
