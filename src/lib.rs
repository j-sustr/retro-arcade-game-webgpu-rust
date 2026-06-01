mod game;
mod geometry;
mod infinite_spawner;
mod input;
mod js_bridge;
mod render;
mod shaders;

use game::{GameMode, GameState};
use geometry::{build_scene_vertices, OBJECT_VERTEX_CAPACITY};
use input::attach_input_handlers;
use js_bridge::{
    await_promise, call_method, console_error_panic_hook, format_js_error, get, object, set,
    show_error,
};
use js_sys::{Array, Promise};
use render::{
    create_bind_group, create_bind_group_layout, create_buffer, create_object_pipeline,
    create_pipeline, look_at, perspective_zo, write_f32_buffer, write_u32_buffer,
    GPU_BUFFER_USAGE_COPY_DST, GPU_BUFFER_USAGE_INDEX, GPU_BUFFER_USAGE_UNIFORM,
    GPU_BUFFER_USAGE_VERTEX, GPU_SHADER_STAGE_FRAGMENT, GPU_SHADER_STAGE_VERTEX,
    GPU_TEXTURE_USAGE_RENDER_ATTACHMENT,
};
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use ui::{create_arcade_overlay, inject_style};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{Document, Element, HtmlCanvasElement, Window};

mod ui;

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
    hud_score: Element,
    hud_shields: Element,
    hud_wave: Element,
    marquee: Element,
    context: JsValue,
    device: JsValue,
    queue: JsValue,
    color_format: String,
    depth_format: &'static str,
    frame_uniform_buffer: JsValue,
    frame_bind_group: JsValue,
    bind_group: JsValue,
    pipeline: JsValue,
    object_pipeline: JsValue,
    vertex_buffer: JsValue,
    index_buffer: JsValue,
    object_buffer: JsValue,
    object_vertex_count: u32,
    depth_texture: Option<JsValue>,
    clear_color: [f32; 4],
    line_color: [f32; 4],
    base_color: [f32; 4],
    line_width: [f32; 2],
    width: u32,
    height: u32,
    last_pointer: Option<(f32, f32)>,
    moving: bool,
    game: GameState,
    last_timestamp: Option<f64>,
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
        let object_pipeline = create_object_pipeline(
            &device,
            &color_format,
            "depth24plus",
            &frame_bind_group_layout,
        )?;

        let vertices: [f32; 20] = [
            -90.0, -0.5, -260.0, 0.0, 0.0, 90.0, -0.5, -260.0, 180.0, 0.0, -90.0, -0.5, 40.0, 0.0,
            300.0, 90.0, -0.5, 40.0, 180.0, 300.0,
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
        let object_buffer = create_buffer(
            &device,
            "Arcade 3D Vertices",
            (OBJECT_VERTEX_CAPACITY * 7 * std::mem::size_of::<f32>()) as u32,
            GPU_BUFFER_USAGE_VERTEX | GPU_BUFFER_USAGE_COPY_DST,
        )?;
        write_f32_buffer(&queue, &vertex_buffer, &vertices)?;
        write_u32_buffer(&queue, &index_buffer, &indices)?;
        let arcade = create_arcade_overlay(&document)?;

        let demo = Rc::new(RefCell::new(Self {
            window,
            document,
            canvas,
            hud_score: arcade.hud_score,
            hud_shields: arcade.hud_shields,
            hud_wave: arcade.hud_wave,
            marquee: arcade.marquee,
            context: context.into(),
            device,
            queue,
            color_format,
            depth_format: "depth24plus",
            frame_uniform_buffer,
            frame_bind_group,
            bind_group,
            pipeline,
            object_pipeline,
            vertex_buffer,
            index_buffer,
            object_buffer,
            object_vertex_count: 0,
            depth_texture: None,
            clear_color: [0.015, 0.0, 0.08, 1.0],
            line_color: [0.0, 0.96, 1.0, 0.92],
            base_color: [0.01, 0.0, 0.035, 1.0],
            line_width: [0.025, 0.025],
            width: 0,
            height: 0,
            last_pointer: None,
            moving: false,
            game: GameState::new(),
            last_timestamp: None,
        }));

        demo.borrow().write_grid_uniforms(&grid_uniform_buffer)?;
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
        let dt = self
            .last_timestamp
            .map(|last| ((timestamp - last) as f32 / 1000.0).clamp(0.0, 0.05))
            .unwrap_or(0.0);
        self.last_timestamp = Some(timestamp);
        self.game.update(dt);
        self.render_hud()?;

        let grid_scroll = self.game.scroll.rem_euclid(12.0) * 6.0;
        let vertices: [f32; 20] = [
            -90.0,
            -0.5,
            -260.0,
            0.0,
            grid_scroll,
            90.0,
            -0.5,
            -260.0,
            180.0,
            grid_scroll,
            -90.0,
            -0.5,
            40.0,
            0.0,
            300.0 + grid_scroll,
            90.0,
            -0.5,
            40.0,
            180.0,
            300.0 + grid_scroll,
        ];
        write_f32_buffer(&self.queue, &self.vertex_buffer, &vertices)?;

        let object_vertices = build_scene_vertices(&self.game);
        self.object_vertex_count = (object_vertices.len() / 7) as u32;
        write_f32_buffer(&self.queue, &self.object_buffer, &object_vertices)?;

        let aspect = self.width as f32 / self.height as f32;
        let projection = perspective_zo(PI * 0.42, aspect, 0.01, 420.0);
        let camera_x = self.game.player.x * 0.18;
        let view = look_at(
            [camera_x, 5.8, 14.0],
            [self.game.player.x * 0.08, -0.2, -42.0],
            [0.0, 1.0, 0.0],
        );
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
        call_method(&pass, "setPipeline", &[self.object_pipeline.clone()])?;
        call_method(
            &pass,
            "setBindGroup",
            &[0.into(), self.frame_bind_group.clone()],
        )?;
        call_method(
            &pass,
            "setVertexBuffer",
            &[0.into(), self.object_buffer.clone()],
        )?;
        call_method(&pass, "draw", &[self.object_vertex_count.into()])?;
        call_method(&pass, "end", &[])?;

        let commands = Array::new();
        commands.push(&call_method(&encoder, "finish", &[])?);
        call_method(&self.queue, "submit", &[commands.into()])?;

        Ok(())
    }

    fn write_grid_uniforms(&self, buffer: &JsValue) -> JsResult<()> {
        let mut uniforms = [0.0; 16];
        uniforms[0..4].copy_from_slice(&self.line_color);
        uniforms[4..8].copy_from_slice(&self.base_color);
        uniforms[8..10].copy_from_slice(&self.line_width);
        write_f32_buffer(&self.queue, buffer, &uniforms)
    }

    fn render_hud(&self) -> JsResult<()> {
        self.hud_score
            .set_text_content(Some(&format!("{:06}", self.game.score)));
        self.hud_shields
            .set_text_content(Some(&format!("{}", self.game.shields)));
        self.hud_wave
            .set_text_content(Some(&format!("{}", self.game.wave)));

        if self.game.mode == GameMode::Attract {
            self.marquee.set_attribute("data-show", "true")?;
            self.marquee.set_text_content(Some("NEON GRID 2084  START"));
        } else if self.game.mode == GameMode::GameOver {
            self.marquee.set_attribute("data-show", "true")?;
            self.marquee.set_text_content(Some("GAME OVER  START"));
        } else if self.game.paused {
            self.marquee.set_attribute("data-show", "true")?;
            self.marquee.set_text_content(Some("PAUSED"));
        } else {
            self.marquee.set_attribute("data-show", "false")?;
        }

        Ok(())
    }
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

fn color(value: &[f32; 4]) -> JsResult<JsValue> {
    let object = object();
    set(&object, "r", value[0])?;
    set(&object, "g", value[1])?;
    set(&object, "b", value[2])?;
    set(&object, "a", value[3])?;
    Ok(object.into())
}
