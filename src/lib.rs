use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect, Uint32Array};
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    Document, Event, HtmlCanvasElement, HtmlInputElement, PointerEvent, WheelEvent, Window,
};

const GRID_SHADER: &str = r#"
fn PristineGrid(uv: vec2f, lineWidth: vec2f) -> f32 {
    let uvDDXY = vec4f(dpdx(uv), dpdy(uv));
    let uvDeriv = vec2f(length(uvDDXY.xz), length(uvDDXY.yw));
    let invertLine: vec2<bool> = lineWidth > vec2f(0.5);
    let targetWidth: vec2f = select(lineWidth, 1 - lineWidth, invertLine);
    let drawWidth: vec2f = clamp(targetWidth, uvDeriv, vec2f(0.5));
    let lineAA: vec2f = uvDeriv * 1.5;
    var gridUV: vec2f = abs(fract(uv) * 2.0 - 1.0);
    gridUV = select(1 - gridUV, gridUV, invertLine);
    var grid2: vec2f = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 *= saturate(targetWidth / drawWidth);
    grid2 = mix(grid2, targetWidth, saturate(uvDeriv * 2.0 - 1.0));
    grid2 = select(grid2, 1.0 - grid2, invertLine);
    return mix(grid2.x, 1.0, grid2.y);
}

struct VertexIn {
  @location(0) pos: vec4f,
  @location(1) uv: vec2f,
}

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

struct Camera {
  projection: mat4x4f,
  view: mat4x4f,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct GridArgs {
  lineColor: vec4f,
  baseColor: vec4f,
  lineWidth: vec2f,
}
@group(1) @binding(0) var<uniform> gridArgs: GridArgs;

@vertex
fn vertexMain(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  out.pos = camera.projection * camera.view * in.pos;
  out.uv = in.uv;
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  let grid = PristineGrid(in.uv, gridArgs.lineWidth);
  return mix(gridArgs.baseColor, gridArgs.lineColor, grid * gridArgs.lineColor.a);
}
"#;

const GPU_BUFFER_USAGE_COPY_DST: u32 = 8;
const GPU_BUFFER_USAGE_INDEX: u32 = 16;
const GPU_BUFFER_USAGE_VERTEX: u32 = 32;
const GPU_BUFFER_USAGE_UNIFORM: u32 = 64;
const GPU_SHADER_STAGE_VERTEX: u32 = 1;
const GPU_SHADER_STAGE_FRAGMENT: u32 = 2;
const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT: u32 = 16;

type JsResult<T> = Result<T, JsValue>;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook();
    spawn_local(async {
        if let Err(error) = GridDemo::start().await {
            show_error(&format_js_error(error));
        }
    });
}

struct GridDemo {
    window: Window,
    document: Document,
    canvas: HtmlCanvasElement,
    context: JsValue,
    device: JsValue,
    queue: JsValue,
    color_format: String,
    depth_format: &'static str,
    frame_uniform_buffer: JsValue,
    frame_bind_group: JsValue,
    bind_group: JsValue,
    pipeline: JsValue,
    vertex_buffer: JsValue,
    index_buffer: JsValue,
    depth_texture: Option<JsValue>,
    clear_color: [f32; 4],
    line_color: [f32; 4],
    base_color: [f32; 4],
    line_width: [f32; 2],
    camera: OrbitCamera,
    width: u32,
    height: u32,
    last_pointer: Option<(f32, f32)>,
    moving: bool,
}

impl GridDemo {
    async fn start() -> JsResult<()> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("Missing window"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("Missing document"))?;
        inject_style(&document)?;

        let canvas = document
            .query_selector(".webgpu-canvas")?
            .and_then(|node| node.dyn_into::<HtmlCanvasElement>().ok())
            .unwrap_or_else(|| {
                let canvas = document
                    .create_element("canvas")
                    .unwrap()
                    .dyn_into::<HtmlCanvasElement>()
                    .unwrap();
                canvas.set_class_name("webgpu-canvas");
                document.body().unwrap().append_child(&canvas).unwrap();
                canvas
            });

        let gpu = get(&window.navigator(), "gpu")?;
        if gpu.is_undefined() {
            return Err(JsValue::from_str(
                "WebGPU is not available in this browser.",
            ));
        }

        let color_format = call_method(&gpu, "getPreferredCanvasFormat", &[])?
            .as_string()
            .unwrap_or_else(|| "bgra8unorm".to_string());
        let adapter =
            await_promise(call_method(&gpu, "requestAdapter", &[])?.dyn_into::<Promise>()?).await?;
        let device =
            await_promise(call_method(&adapter, "requestDevice", &[])?.dyn_into::<Promise>()?)
                .await?;
        let queue = get(&device, "queue")?;
        let context = canvas
            .get_context("webgpu")?
            .ok_or_else(|| JsValue::from_str("Unable to create a WebGPU canvas context"))?;

        let config = object();
        set(&config, "device", &device)?;
        set(&config, "format", &color_format)?;
        set(&config, "alphaMode", "opaque")?;
        let view_formats = Array::new();
        view_formats.push(&JsValue::from_str(&format!("{color_format}-srgb")));
        set(&config, "viewFormats", &view_formats)?;
        call_method(&context, "configure", &[config.into()])?;

        let frame_bind_group_layout = create_bind_group_layout(
            &device,
            "Frame BindGroupLayout",
            GPU_SHADER_STAGE_VERTEX | GPU_SHADER_STAGE_FRAGMENT,
        )?;
        let grid_bind_group_layout =
            create_bind_group_layout(&device, "Grid BindGroupLayout", GPU_SHADER_STAGE_FRAGMENT)?;

        let frame_uniform_buffer = create_buffer(
            &device,
            "Frame Uniforms",
            32 * std::mem::size_of::<f32>() as u32,
            GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
        )?;
        let grid_uniform_buffer = create_buffer(
            &device,
            "Grid Uniforms",
            16 * std::mem::size_of::<f32>() as u32,
            GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
        )?;

        let frame_bind_group = create_bind_group(
            &device,
            "Frame BindGroup",
            &frame_bind_group_layout,
            &frame_uniform_buffer,
        )?;
        let bind_group = create_bind_group(
            &device,
            "Grid BindGroup",
            &grid_bind_group_layout,
            &grid_uniform_buffer,
        )?;

        let pipeline = create_pipeline(
            &device,
            &color_format,
            "depth24plus",
            &frame_bind_group_layout,
            &grid_bind_group_layout,
        )?;

        let vertices: [f32; 20] = [
            -20.0, -0.5, -20.0, 0.0, 0.0, 20.0, -0.5, -20.0, 200.0, 0.0, -20.0, -0.5, 20.0, 0.0,
            200.0, 20.0, -0.5, 20.0, 200.0, 200.0,
        ];
        let indices: [u32; 6] = [0, 1, 2, 1, 2, 3];
        let vertex_buffer = create_buffer(
            &device,
            "Grid Vertices",
            (vertices.len() * std::mem::size_of::<f32>()) as u32,
            GPU_BUFFER_USAGE_VERTEX | GPU_BUFFER_USAGE_COPY_DST,
        )?;
        let index_buffer = create_buffer(
            &device,
            "Grid Indices",
            (indices.len() * std::mem::size_of::<u32>()) as u32,
            GPU_BUFFER_USAGE_INDEX | GPU_BUFFER_USAGE_COPY_DST,
        )?;
        write_f32_buffer(&queue, &vertex_buffer, &vertices)?;
        write_u32_buffer(&queue, &index_buffer, &indices)?;

        let demo = Rc::new(RefCell::new(Self {
            window,
            document,
            canvas,
            context: context.into(),
            device,
            queue,
            color_format,
            depth_format: "depth24plus",
            frame_uniform_buffer,
            frame_bind_group,
            bind_group,
            pipeline,
            vertex_buffer,
            index_buffer,
            depth_texture: None,
            clear_color: [0.0, 0.0, 0.2, 1.0],
            line_color: [1.0, 1.0, 1.0, 1.0],
            base_color: [0.0, 0.0, 0.0, 1.0],
            line_width: [0.05, 0.05],
            camera: OrbitCamera::new(),
            width: 0,
            height: 0,
            last_pointer: None,
            moving: false,
        }));

        demo.borrow().write_grid_uniforms(&grid_uniform_buffer)?;
        create_controls(&demo, &grid_uniform_buffer)?;
        attach_input_handlers(&demo)?;
        attach_resize_handler(&demo)?;
        demo.borrow_mut().resize()?;
        animate(demo)?;
        Ok(())
    }

    fn resize(&mut self) -> JsResult<()> {
        let dpr = self.window.device_pixel_ratio().max(1.0);
        let width = ((self.canvas.client_width() as f64) * dpr).max(1.0) as u32;
        let height = ((self.canvas.client_height() as f64) * dpr).max(1.0) as u32;
        if width == self.width && height == self.height {
            return Ok(());
        }

        self.width = width;
        self.height = height;
        self.canvas.set_width(width);
        self.canvas.set_height(height);

        if let Some(texture) = self.depth_texture.take() {
            let _ = call_method(&texture, "destroy", &[]);
        }

        let desc = object();
        let size = object();
        set(&size, "width", width)?;
        set(&size, "height", height)?;
        set(&desc, "size", &size)?;
        set(&desc, "format", self.depth_format)?;
        set(&desc, "usage", GPU_TEXTURE_USAGE_RENDER_ATTACHMENT)?;
        self.depth_texture = Some(call_method(&self.device, "createTexture", &[desc.into()])?);
        Ok(())
    }

    fn frame(&mut self, timestamp: f64) -> JsResult<()> {
        self.resize()?;

        let aspect = self.width as f32 / self.height as f32;
        let projection = perspective_zo(PI * 0.5, aspect, 0.01, 128.0);
        let view = self.camera.view_matrix();
        let mut frame_uniforms = [0.0; 32];
        frame_uniforms[0..16].copy_from_slice(&projection);
        frame_uniforms[16..32].copy_from_slice(&view);
        write_f32_buffer(&self.queue, &self.frame_uniform_buffer, &frame_uniforms)?;

        let surface = call_method(&self.context, "getCurrentTexture", &[])?;
        let surface_view_desc = object();
        set(
            &surface_view_desc,
            "format",
            format!("{}-srgb", self.color_format),
        )?;
        let surface_view = call_method(&surface, "createView", &[surface_view_desc.into()])?;

        let depth_view = call_method(
            self.depth_texture
                .as_ref()
                .ok_or_else(|| JsValue::from_str("Missing depth texture"))?,
            "createView",
            &[],
        )?;

        let color_attachment = object();
        set(&color_attachment, "view", &surface_view)?;
        set(&color_attachment, "clearValue", color(&self.clear_color)?)?;
        set(&color_attachment, "loadOp", "clear")?;
        set(&color_attachment, "storeOp", "store")?;
        let color_attachments = Array::new();
        color_attachments.push(&color_attachment);

        let depth_attachment = object();
        set(&depth_attachment, "view", &depth_view)?;
        set(&depth_attachment, "depthClearValue", 1.0)?;
        set(&depth_attachment, "depthLoadOp", "clear")?;
        set(&depth_attachment, "depthStoreOp", "discard")?;

        let pass_desc = object();
        set(&pass_desc, "colorAttachments", &color_attachments)?;
        set(&pass_desc, "depthStencilAttachment", &depth_attachment)?;

        let encoder = call_method(&self.device, "createCommandEncoder", &[])?;
        let pass = call_method(&encoder, "beginRenderPass", &[pass_desc.into()])?;
        call_method(&pass, "setPipeline", &[self.pipeline.clone()])?;
        call_method(
            &pass,
            "setBindGroup",
            &[0.into(), self.frame_bind_group.clone()],
        )?;
        call_method(&pass, "setBindGroup", &[1.into(), self.bind_group.clone()])?;
        call_method(
            &pass,
            "setVertexBuffer",
            &[0.into(), self.vertex_buffer.clone()],
        )?;
        call_method(
            &pass,
            "setIndexBuffer",
            &[self.index_buffer.clone(), "uint32".into()],
        )?;
        call_method(&pass, "drawIndexed", &[6.into()])?;
        call_method(&pass, "end", &[])?;

        let commands = Array::new();
        commands.push(&call_method(&encoder, "finish", &[])?);
        call_method(&self.queue, "submit", &[commands.into()])?;

        if let Some(node) = self.document.get_element_by_id("frame-time") {
            node.set_text_content(Some(&format!("{:.2} ms", timestamp.fract())));
        }
        Ok(())
    }

    fn write_grid_uniforms(&self, buffer: &JsValue) -> JsResult<()> {
        let mut uniforms = [0.0; 16];
        uniforms[0..4].copy_from_slice(&self.line_color);
        uniforms[4..8].copy_from_slice(&self.base_color);
        uniforms[8..10].copy_from_slice(&self.line_width);
        write_f32_buffer(&self.queue, buffer, &uniforms)
    }
}

fn create_controls(demo: &Rc<RefCell<GridDemo>>, uniform_buffer: &JsValue) -> JsResult<()> {
    let document = demo.borrow().document.clone();
    let panel = document.create_element("form")?;
    panel.set_class_name("controls");
    panel.set_inner_html(
        r##"
        <h1>Pristine Grid</h1>
        <label>Clear <input id="clear-color" type="color" value="#000033"></label>
        <label>Base <input id="base-color" type="color" value="#000000"></label>
        <label>Line <input id="line-color" type="color" value="#ffffff"></label>
        <label>Line Alpha <input id="line-alpha" type="range" min="0" max="1" step="0.01" value="1"></label>
        <label>Width X <input id="line-width-x" type="range" min="0" max="1" step="0.001" value="0.05"></label>
        <label>Width Y <input id="line-width-y" type="range" min="0" max="1" step="0.001" value="0.05"></label>
        <a href="https://github.com/toji/pristine-grid-webgpu" target="_blank" rel="noreferrer">View Source</a>
        "##,
    );
    document.body().unwrap().append_child(&panel)?;

    bind_color(demo, uniform_buffer, "clear-color", |demo, color| {
        demo.clear_color = color
    })?;
    bind_color(demo, uniform_buffer, "base-color", |demo, color| {
        demo.base_color = color
    })?;
    bind_color(demo, uniform_buffer, "line-color", |demo, color| {
        demo.line_color[0] = color[0];
        demo.line_color[1] = color[1];
        demo.line_color[2] = color[2];
    })?;
    bind_range(demo, uniform_buffer, "line-alpha", |demo, value| {
        demo.line_color[3] = value
    })?;
    bind_range(demo, uniform_buffer, "line-width-x", |demo, value| {
        demo.line_width[0] = value
    })?;
    bind_range(demo, uniform_buffer, "line-width-y", |demo, value| {
        demo.line_width[1] = value
    })?;
    Ok(())
}

fn bind_color<F>(
    demo: &Rc<RefCell<GridDemo>>,
    uniform_buffer: &JsValue,
    id: &str,
    mut update: F,
) -> JsResult<()>
where
    F: 'static + FnMut(&mut GridDemo, [f32; 4]),
{
    let input = input_by_id(&demo.borrow().document, id)?;
    let demo = Rc::clone(demo);
    let uniform_buffer = uniform_buffer.clone();
    let closure = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
        let input: HtmlInputElement = event.target().unwrap().dyn_into().unwrap();
        let color = parse_hex_color(&input.value(), 1.0);
        let mut demo = demo.borrow_mut();
        update(&mut demo, color);
        let _ = demo.write_grid_uniforms(&uniform_buffer);
    });
    input.add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
    closure.forget();
    Ok(())
}

fn bind_range<F>(
    demo: &Rc<RefCell<GridDemo>>,
    uniform_buffer: &JsValue,
    id: &str,
    mut update: F,
) -> JsResult<()>
where
    F: 'static + FnMut(&mut GridDemo, f32),
{
    let input = input_by_id(&demo.borrow().document, id)?;
    let demo = Rc::clone(demo);
    let uniform_buffer = uniform_buffer.clone();
    let closure = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
        let input: HtmlInputElement = event.target().unwrap().dyn_into().unwrap();
        let value = input.value().parse::<f32>().unwrap_or(0.0);
        let demo_ref = &mut *demo.borrow_mut();
        update(demo_ref, value);
        let _ = demo_ref.write_grid_uniforms(&uniform_buffer);
    });
    input.add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
    closure.forget();
    Ok(())
}

fn attach_input_handlers(demo: &Rc<RefCell<GridDemo>>) -> JsResult<()> {
    let canvas = demo.borrow().canvas.clone();

    let down_demo = Rc::clone(demo);
    let on_down = Closure::<dyn FnMut(PointerEvent)>::new(move |event: PointerEvent| {
        let mut demo = down_demo.borrow_mut();
        demo.moving = event.is_primary();
        demo.last_pointer = Some((event.page_x() as f32, event.page_y() as f32));
    });
    canvas.add_event_listener_with_callback("pointerdown", on_down.as_ref().unchecked_ref())?;
    on_down.forget();

    let move_demo = Rc::clone(demo);
    let on_move = Closure::<dyn FnMut(PointerEvent)>::new(move |event: PointerEvent| {
        let mut demo = move_demo.borrow_mut();
        if !demo.moving {
            return;
        }
        let current = (event.page_x() as f32, event.page_y() as f32);
        if let Some(last) = demo.last_pointer {
            demo.camera
                .orbit((current.0 - last.0) * 0.025, (current.1 - last.1) * 0.025);
        }
        demo.last_pointer = Some(current);
    });
    canvas.add_event_listener_with_callback("pointermove", on_move.as_ref().unchecked_ref())?;
    on_move.forget();

    let up_demo = Rc::clone(demo);
    let on_up = Closure::<dyn FnMut(PointerEvent)>::new(move |event: PointerEvent| {
        if event.is_primary() {
            up_demo.borrow_mut().moving = false;
        }
    });
    canvas.add_event_listener_with_callback("pointerup", on_up.as_ref().unchecked_ref())?;
    canvas.add_event_listener_with_callback("pointercancel", on_up.as_ref().unchecked_ref())?;
    on_up.forget();

    let wheel_demo = Rc::clone(demo);
    let on_wheel = Closure::<dyn FnMut(WheelEvent)>::new(move |event: WheelEvent| {
        event.prevent_default();
        let mut demo = wheel_demo.borrow_mut();
        let distance = demo.camera.distance + event.delta_y() as f32 * 0.005;
        demo.camera.distance = distance.clamp(1.0, 10.0);
        demo.camera.dirty = true;
    });
    canvas.add_event_listener_with_callback("wheel", on_wheel.as_ref().unchecked_ref())?;
    on_wheel.forget();

    Ok(())
}

fn attach_resize_handler(demo: &Rc<RefCell<GridDemo>>) -> JsResult<()> {
    let window = demo.borrow().window.clone();
    let demo = Rc::clone(demo);
    let on_resize = Closure::<dyn FnMut()>::new(move || {
        let _ = demo.borrow_mut().resize();
    });
    window.add_event_listener_with_callback("resize", on_resize.as_ref().unchecked_ref())?;
    on_resize.forget();
    Ok(())
}

fn animate(demo: Rc<RefCell<GridDemo>>) -> JsResult<()> {
    let callback = Rc::new(RefCell::new(None::<Closure<dyn FnMut(f64)>>));
    let callback_ref = Rc::clone(&callback);
    let window = demo.borrow().window.clone();
    let animation_window = window.clone();

    *callback_ref.borrow_mut() = Some(Closure::wrap(Box::new(move |timestamp: f64| {
        if let Err(error) = demo.borrow_mut().frame(timestamp) {
            show_error(&format_js_error(error));
        }
        let _ = animation_window
            .request_animation_frame(callback.borrow().as_ref().unwrap().as_ref().unchecked_ref());
    }) as Box<dyn FnMut(f64)>));

    window.request_animation_frame(
        callback_ref
            .borrow()
            .as_ref()
            .unwrap()
            .as_ref()
            .unchecked_ref(),
    )?;
    Ok(())
}

struct OrbitCamera {
    orbit_x: f32,
    orbit_y: f32,
    distance: f32,
    dirty: bool,
    view: [f32; 16],
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            orbit_x: 0.0,
            orbit_y: 0.0,
            distance: 1.0,
            dirty: true,
            view: identity(),
        }
    }

    fn orbit(&mut self, x_delta: f32, y_delta: f32) {
        self.orbit_y = wrap_pi(self.orbit_y + x_delta);
        self.orbit_x = (self.orbit_x + y_delta).clamp(-PI * 0.5, PI * 0.5);
        self.dirty = true;
    }

    fn view_matrix(&mut self) -> [f32; 16] {
        if self.dirty {
            let camera = multiply(
                &multiply(&rotation_y(-self.orbit_y), &rotation_x(-self.orbit_x)),
                &translation(0.0, 0.0, self.distance),
            );
            self.view = invert_rigid_body(&camera);
            self.dirty = false;
        }
        self.view
    }
}

fn create_pipeline(
    device: &JsValue,
    color_format: &str,
    depth_format: &str,
    frame_layout: &JsValue,
    grid_layout: &JsValue,
) -> JsResult<JsValue> {
    let module_desc = object();
    set(&module_desc, "label", "Pristine Grid")?;
    set(&module_desc, "code", GRID_SHADER)?;
    let module = call_method(device, "createShaderModule", &[module_desc.into()])?;

    let layouts = Array::new();
    layouts.push(frame_layout);
    layouts.push(grid_layout);
    let pipeline_layout_desc = object();
    set(&pipeline_layout_desc, "bindGroupLayouts", &layouts)?;
    let pipeline_layout = call_method(
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
    call_method(device, "createRenderPipeline", &[desc.into()])
}

fn create_bind_group_layout(device: &JsValue, label: &str, visibility: u32) -> JsResult<JsValue> {
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
    call_method(device, "createBindGroupLayout", &[desc.into()])
}

fn create_bind_group(
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
    call_method(device, "createBindGroup", &[desc.into()])
}

fn create_buffer(device: &JsValue, label: &str, size: u32, usage: u32) -> JsResult<JsValue> {
    let desc = object();
    set(&desc, "label", label)?;
    set(&desc, "size", size)?;
    set(&desc, "usage", usage)?;
    call_method(device, "createBuffer", &[desc.into()])
}

fn write_f32_buffer(queue: &JsValue, buffer: &JsValue, data: &[f32]) -> JsResult<()> {
    let array = Float32Array::from(data);
    call_method(
        queue,
        "writeBuffer",
        &[buffer.clone(), 0.into(), array.into()],
    )?;
    Ok(())
}

fn write_u32_buffer(queue: &JsValue, buffer: &JsValue, data: &[u32]) -> JsResult<()> {
    let array = Uint32Array::from(data);
    call_method(
        queue,
        "writeBuffer",
        &[buffer.clone(), 0.into(), array.into()],
    )?;
    Ok(())
}

fn perspective_zo(fovy: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
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

fn identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn translation(x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
    ]
}

fn rotation_x(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    [
        1.0, 0.0, 0.0, 0.0, 0.0, c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn rotation_y(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    [
        c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn multiply(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    out
}

fn invert_rigid_body(m: &[f32; 16]) -> [f32; 16] {
    let mut out = identity();
    out[0] = m[0];
    out[1] = m[4];
    out[2] = m[8];
    out[4] = m[1];
    out[5] = m[5];
    out[6] = m[9];
    out[8] = m[2];
    out[9] = m[6];
    out[10] = m[10];
    out[12] = -(out[0] * m[12] + out[4] * m[13] + out[8] * m[14]);
    out[13] = -(out[1] * m[12] + out[5] * m[13] + out[9] * m[14]);
    out[14] = -(out[2] * m[12] + out[6] * m[13] + out[10] * m[14]);
    out
}

fn wrap_pi(mut value: f32) -> f32 {
    while value < -PI {
        value += PI * 2.0;
    }
    while value >= PI {
        value -= PI * 2.0;
    }
    value
}

fn parse_hex_color(hex: &str, alpha: f32) -> [f32; 4] {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return [0.0, 0.0, 0.0, alpha];
    }
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0) as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0) as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0) as f32 / 255.0;
    [r, g, b, alpha]
}

fn input_by_id(document: &Document, id: &str) -> JsResult<HtmlInputElement> {
    document
        .get_element_by_id(id)
        .ok_or_else(|| JsValue::from_str(&format!("Missing input #{id}")))?
        .dyn_into::<HtmlInputElement>()
        .map_err(|_| JsValue::from_str(&format!("#{id} is not an input")))
}

fn inject_style(document: &Document) -> JsResult<()> {
    let style = document.create_element("style")?;
    style.set_text_content(Some(
        r#"
        html, body {
          height: 100%;
          margin: 0;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #151515;
          overflow: hidden;
        }

        canvas {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          touch-action: none;
        }

        .controls {
          position: fixed;
          top: 14px;
          right: 14px;
          z-index: 3;
          width: min(280px, calc(100vw - 28px));
          display: grid;
          gap: 10px;
          padding: 12px;
          box-sizing: border-box;
          background: rgba(24, 24, 24, 0.88);
          border: 1px solid rgba(255, 255, 255, 0.14);
          border-radius: 8px;
          color: #f3f3f3;
          backdrop-filter: blur(12px);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
        }

        .controls h1 {
          margin: 0 0 2px;
          font-size: 15px;
          font-weight: 650;
          letter-spacing: 0;
        }

        .controls label {
          display: grid;
          grid-template-columns: 88px minmax(0, 1fr);
          align-items: center;
          gap: 10px;
          font-size: 12px;
          color: #d8d8d8;
        }

        .controls input {
          width: 100%;
          box-sizing: border-box;
        }

        .controls input[type="color"] {
          height: 28px;
          border: 0;
          padding: 0;
          background: transparent;
        }

        .controls a {
          color: #9fd3ff;
          font-size: 12px;
          text-decoration: none;
        }

        .error {
          position: fixed;
          z-index: 4;
          inset: 4rem;
          color: #ffb2b2;
          white-space: pre-wrap;
        }
        "#,
    ));
    document
        .query_selector("head")?
        .ok_or_else(|| JsValue::from_str("Missing document head"))?
        .append_child(&style)?;
    Ok(())
}

fn color(value: &[f32; 4]) -> JsResult<JsValue> {
    let object = object();
    set(&object, "r", value[0])?;
    set(&object, "g", value[1])?;
    set(&object, "b", value[2])?;
    set(&object, "a", value[3])?;
    Ok(object.into())
}

async fn await_promise(promise: Promise) -> JsResult<JsValue> {
    JsFuture::from(promise).await
}

fn object() -> Object {
    Object::new()
}

fn get<T: AsRef<JsValue>>(object: &T, key: &str) -> JsResult<JsValue> {
    Reflect::get(object.as_ref(), &JsValue::from_str(key))
}

fn set<T: Into<JsValue>>(object: &Object, key: &str, value: T) -> JsResult<()> {
    Reflect::set(object, &JsValue::from_str(key), &value.into()).map(|_| ())
}

fn call_method<T: AsRef<JsValue>>(object: &T, key: &str, args: &[JsValue]) -> JsResult<JsValue> {
    let function = get(object, key)?.dyn_into::<Function>()?;
    let args_array = Array::new();
    for arg in args {
        args_array.push(arg);
    }
    Reflect::apply(&function, object.as_ref(), &args_array)
}

fn show_error(message: &str) {
    if let Some(document) = web_sys::window().and_then(|window| window.document()) {
        if let Ok(error) = document.create_element("pre") {
            error.set_class_name("error");
            error.set_text_content(Some(message));
            let _ = document.body().unwrap().append_child(&error);
        }
    }
    web_sys::console::error_1(&JsValue::from_str(message));
}

fn format_js_error(error: JsValue) -> String {
    if let Some(message) = error.as_string() {
        message
    } else {
        format!("{error:?}")
    }
}

fn console_error_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
    }));
}
