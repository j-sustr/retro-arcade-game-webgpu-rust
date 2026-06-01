mod infinite_spawner;

use infinite_spawner::{ActiveMountain, ActiveObject, InfiniteSpawner};
use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect, Uint32Array};
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{Document, Element, HtmlCanvasElement, KeyboardEvent, PointerEvent, Window};

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

const OBJECT_SHADER: &str = r#"
struct VertexIn {
  @location(0) pos: vec4f,
  @location(1) color: vec4f,
}

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
}

struct Camera {
  projection: mat4x4f,
  view: mat4x4f,
}
@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vertexMain(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  out.pos = camera.projection * camera.view * in.pos;
  out.color = in.color;
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  return in.color;
}
"#;

const GPU_BUFFER_USAGE_COPY_DST: u32 = 8;
const GPU_BUFFER_USAGE_INDEX: u32 = 16;
const GPU_BUFFER_USAGE_VERTEX: u32 = 32;
const GPU_BUFFER_USAGE_UNIFORM: u32 = 64;
const GPU_SHADER_STAGE_VERTEX: u32 = 1;
const GPU_SHADER_STAGE_FRAGMENT: u32 = 2;
const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT: u32 = 16;
const OBJECT_VERTEX_CAPACITY: usize = 24_000;

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum GameMode {
    Attract,
    Playing,
    GameOver,
}

struct ArcadeOverlay {
    hud_score: Element,
    hud_shields: Element,
    hud_wave: Element,
    marquee: Element,
}

struct GameState {
    mode: GameMode,
    paused: bool,
    player: Player,
    spawner: InfiniteSpawner,
    input: InputState,
    score: u32,
    shields: u32,
    wave: u32,
    scroll: f32,
    speed: f32,
    rng: u32,
}

struct Player {
    x: f32,
    tilt: f32,
}

#[derive(Default)]
struct InputState {
    left: bool,
    right: bool,
    boost: bool,
}

impl GameState {
    fn new() -> Self {
        Self {
            mode: GameMode::Attract,
            paused: false,
            player: Player { x: 0.0, tilt: 0.0 },
            spawner: InfiniteSpawner::new(),
            input: InputState::default(),
            score: 0,
            shields: 5,
            wave: 1,
            scroll: 0.0,
            speed: 42.0,
            rng: 0x2084_1986,
        }
    }

    fn start(&mut self) {
        let rng = self.rng.wrapping_add(0x9e37_79b9);
        *self = Self::new();
        self.rng = rng;
        self.spawner.reset(rng);
        self.mode = GameMode::Playing;
    }

    fn update(&mut self, dt: f32) {
        if self.mode != GameMode::Playing || self.paused {
            return;
        }

        let mut dx: f32 = 0.0;
        if self.input.left {
            dx -= 1.0;
        }
        if self.input.right {
            dx += 1.0;
        }

        self.speed = 42.0 + self.wave as f32 * 4.5 + if self.input.boost { 18.0 } else { 0.0 };
        self.scroll += self.speed * dt;
        self.player.x = (self.player.x + dx * 24.0 * dt).clamp(-13.0, 13.0);
        self.player.tilt += ((dx * 18.0) - self.player.tilt) * (dt * 12.0).min(1.0);

        self.spawner.update(self.scroll, self.wave);
        self.resolve_hits();

        self.score = self.score.saturating_add((dt * self.speed * 2.0) as u32);
        self.wave = 1 + self.score / 1600;
    }

    fn resolve_hits(&mut self) {
        let px = self.player.x;
        if self.spawner.hit_obstacle(px, self.scroll) {
            self.shields = self.shields.saturating_sub(1);
            if self.shields == 0 {
                self.mode = GameMode::GameOver;
            }
        }

        if self.spawner.collect_pickup(px, self.scroll) {
            self.score = self.score.saturating_add(500);
            self.shields = (self.shields + 1).min(5);
        }
    }
}

fn build_scene_vertices(game: &GameState) -> Vec<f32> {
    let mut out = Vec::with_capacity(OBJECT_VERTEX_CAPACITY * 7);
    add_rider(&mut out, game.player.x, game.player.tilt);

    for obstacle in game.spawner.obstacles(game.scroll) {
        if obstacle.kind == 0 {
            add_obstacle_pyramid(&mut out, obstacle, [1.0, 0.05, 0.78, 1.0]);
        } else {
            add_obstacle_box(&mut out, obstacle, [0.0, 0.95, 1.0, 1.0]);
        }
    }

    for pickup in game.spawner.pickups(game.scroll) {
        add_pickup_gate(&mut out, pickup.x, pickup.z);
    }

    for mountain in game.spawner.mountains(game.scroll) {
        let color = if mountain.kind == 0 {
            [0.16, 0.0, 0.38, 1.0]
        } else {
            [0.08, 0.0, 0.3, 1.0]
        };
        add_mountain(&mut out, mountain, color);
    }

    out.truncate(OBJECT_VERTEX_CAPACITY * 7);
    out
}

fn add_rider(out: &mut Vec<f32>, x: f32, tilt: f32) {
    let z = 4.0;
    let lean = tilt.to_radians().sin() * 0.65;
    add_triangle(
        out,
        [x + lean, 1.45, z - 3.2],
        [x - 1.45, -0.22, z + 2.0],
        [x + 1.45, -0.22, z + 2.0],
        [0.0, 0.96, 1.0, 1.0],
    );
    add_triangle(
        out,
        [x + lean, 1.45, z - 3.2],
        [x + 1.45, -0.22, z + 2.0],
        [x, -0.4, z + 3.4],
        [1.0, 0.12, 0.8, 1.0],
    );
    add_triangle(
        out,
        [x + lean, 1.45, z - 3.2],
        [x, -0.4, z + 3.4],
        [x - 1.45, -0.22, z + 2.0],
        [0.42, 0.2, 1.0, 1.0],
    );
    add_box(
        out,
        x,
        0.05,
        z + 1.7,
        0.42,
        0.18,
        1.4,
        [1.0, 0.95, 0.0, 1.0],
    );
}

fn add_obstacle_box(out: &mut Vec<f32>, obstacle: ActiveObject, color: [f32; 4]) {
    add_box(
        out,
        obstacle.x,
        obstacle.size * 0.55,
        obstacle.z,
        obstacle.size * 0.75,
        obstacle.size * 1.1,
        obstacle.size * 0.75,
        color,
    );
}

fn add_obstacle_pyramid(out: &mut Vec<f32>, obstacle: ActiveObject, color: [f32; 4]) {
    let y = -0.48;
    let s = obstacle.size * 0.95;
    let x = obstacle.x;
    let z = obstacle.z;
    let p0 = [x - s, y, z - s];
    let p1 = [x + s, y, z - s];
    let p2 = [x + s, y, z + s];
    let p3 = [x - s, y, z + s];
    let top = [x, y + obstacle.size * 2.0, z];
    add_triangle(out, p0, p1, top, color);
    add_triangle(out, p1, p2, top, [0.0, 0.9, 1.0, 1.0]);
    add_triangle(out, p2, p3, top, color);
    add_triangle(out, p3, p0, top, [0.5, 0.08, 1.0, 1.0]);
}

fn add_pickup_gate(out: &mut Vec<f32>, x: f32, z: f32) {
    add_box(out, x - 1.25, 1.0, z, 0.2, 1.7, 0.2, [1.0, 1.0, 0.0, 1.0]);
    add_box(out, x + 1.25, 1.0, z, 0.2, 1.7, 0.2, [1.0, 1.0, 0.0, 1.0]);
    add_box(out, x, 2.65, z, 1.45, 0.2, 0.2, [1.0, 0.1, 0.82, 1.0]);
}

fn add_mountain(out: &mut Vec<f32>, mountain: ActiveMountain, color: [f32; 4]) {
    let base_y = -0.55;
    let x = mountain.x;
    let z = mountain.z;
    let width = mountain.width;
    let height = mountain.height;
    let p0 = [x - width, base_y, z + 10.0];
    let p1 = [x + width, base_y, z + 10.0];
    let p2 = [x, base_y, z - width * 0.55];
    let top = [x + width * 0.1, height, z + 1.0];
    add_triangle(out, p0, p1, top, color);
    add_triangle(out, p1, p2, top, [0.0, 0.35, 0.62, 1.0]);
    add_triangle(out, p2, p0, top, [0.52, 0.02, 0.58, 1.0]);
}

fn add_box(out: &mut Vec<f32>, x: f32, y: f32, z: f32, hx: f32, hy: f32, hz: f32, color: [f32; 4]) {
    let p = [
        [x - hx, y - hy, z - hz],
        [x + hx, y - hy, z - hz],
        [x + hx, y + hy, z - hz],
        [x - hx, y + hy, z - hz],
        [x - hx, y - hy, z + hz],
        [x + hx, y - hy, z + hz],
        [x + hx, y + hy, z + hz],
        [x - hx, y + hy, z + hz],
    ];
    add_quad(out, p[0], p[1], p[2], p[3], color);
    add_quad(out, p[5], p[4], p[7], p[6], color);
    add_quad(
        out,
        p[4],
        p[0],
        p[3],
        p[7],
        [color[0] * 0.65, color[1] * 0.65, color[2], 1.0],
    );
    add_quad(
        out,
        p[1],
        p[5],
        p[6],
        p[2],
        [color[0], color[1] * 0.72, color[2] * 0.72, 1.0],
    );
    add_quad(out, p[3], p[2], p[6], p[7], [1.0, 1.0, 1.0, 1.0]);
}

fn add_quad(
    out: &mut Vec<f32>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    color: [f32; 4],
) {
    add_triangle(out, a, b, c, color);
    add_triangle(out, a, c, d, color);
}

fn add_triangle(out: &mut Vec<f32>, a: [f32; 3], b: [f32; 3], c: [f32; 3], color: [f32; 4]) {
    for p in [a, b, c] {
        out.extend_from_slice(&[p[0], p[1], p[2], color[0], color[1], color[2], color[3]]);
    }
}

fn create_arcade_overlay(document: &Document) -> JsResult<ArcadeOverlay> {
    let hud = document.create_element("section")?;
    hud.set_class_name("hud");
    hud.set_inner_html(
        r#"
        <div><span>SCORE</span><strong id="hud-score">000000</strong></div>
        <div><span>SHIELDS</span><strong id="hud-shields">3</strong></div>
        <div><span>WAVE</span><strong id="hud-wave">1</strong></div>
        "#,
    );
    document.body().unwrap().append_child(&hud)?;

    let marquee = document.create_element("div")?;
    marquee.set_class_name("marquee");
    marquee.set_attribute("data-show", "true")?;
    marquee.set_text_content(Some("NEON GRID 2084  START"));
    document.body().unwrap().append_child(&marquee)?;

    Ok(ArcadeOverlay {
        hud_score: required_element(document, "hud-score")?,
        hud_shields: required_element(document, "hud-shields")?,
        hud_wave: required_element(document, "hud-wave")?,
        marquee,
    })
}

fn attach_input_handlers(demo: &Rc<RefCell<GridDemo>>) -> JsResult<()> {
    let canvas = demo.borrow().canvas.clone();

    let key_demo = Rc::clone(demo);
    let on_key_down = Closure::<dyn FnMut(KeyboardEvent)>::new(move |event: KeyboardEvent| {
        handle_key(&key_demo, &event, true);
    });
    demo.borrow()
        .document
        .add_event_listener_with_callback("keydown", on_key_down.as_ref().unchecked_ref())?;
    on_key_down.forget();

    let key_demo = Rc::clone(demo);
    let on_key_up = Closure::<dyn FnMut(KeyboardEvent)>::new(move |event: KeyboardEvent| {
        handle_key(&key_demo, &event, false);
    });
    demo.borrow()
        .document
        .add_event_listener_with_callback("keyup", on_key_up.as_ref().unchecked_ref())?;
    on_key_up.forget();

    let down_demo = Rc::clone(demo);
    let on_down = Closure::<dyn FnMut(PointerEvent)>::new(move |event: PointerEvent| {
        let mut demo = down_demo.borrow_mut();
        demo.moving = event.is_primary();
        demo.last_pointer = Some((event.page_x() as f32, event.page_y() as f32));
        if demo.game.mode != GameMode::Playing {
            demo.game.start();
        }
        demo.game.input.boost = true;
        position_player_from_pointer(&mut demo, event.client_x() as f32, event.client_y() as f32);
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
        position_player_from_pointer(&mut demo, event.client_x() as f32, event.client_y() as f32);
        demo.player_tilt_from_pointer(current.0);
        demo.last_pointer = Some(current);
    });
    canvas.add_event_listener_with_callback("pointermove", on_move.as_ref().unchecked_ref())?;
    on_move.forget();

    let up_demo = Rc::clone(demo);
    let on_up = Closure::<dyn FnMut(PointerEvent)>::new(move |event: PointerEvent| {
        if event.is_primary() {
            let mut demo = up_demo.borrow_mut();
            demo.moving = false;
            demo.game.input.boost = false;
        }
    });
    canvas.add_event_listener_with_callback("pointerup", on_up.as_ref().unchecked_ref())?;
    canvas.add_event_listener_with_callback("pointercancel", on_up.as_ref().unchecked_ref())?;
    on_up.forget();

    Ok(())
}

impl GridDemo {
    fn player_tilt_from_pointer(&mut self, x: f32) {
        if let Some(last) = self.last_pointer {
            self.game.player.tilt = ((x - last.0) * 0.8).clamp(-24.0, 24.0);
        }
    }
}

fn handle_key(demo: &Rc<RefCell<GridDemo>>, event: &KeyboardEvent, pressed: bool) {
    let key = event.key();
    let mut demo = demo.borrow_mut();
    match key.as_str() {
        "ArrowLeft" | "a" | "A" => demo.game.input.left = pressed,
        "ArrowRight" | "d" | "D" => demo.game.input.right = pressed,
        "ArrowUp" | "w" | "W" => demo.game.input.boost = pressed,
        "ArrowDown" | "s" | "S" => demo.game.input.boost = false,
        " " | "Spacebar" => {
            if pressed {
                if demo.game.mode != GameMode::Playing {
                    demo.game.start();
                } else {
                    demo.game.input.boost = true;
                }
            } else {
                demo.game.input.boost = false;
            }
        }
        "Enter" => {
            if pressed && demo.game.mode != GameMode::Playing {
                demo.game.start();
            }
        }
        "p" | "P" => {
            if pressed && demo.game.mode == GameMode::Playing {
                demo.game.paused = !demo.game.paused;
            }
        }
        _ => return,
    }
    event.prevent_default();
}

fn position_player_from_pointer(demo: &mut GridDemo, client_x: f32, _client_y: f32) {
    let width = demo.canvas.client_width().max(1) as f32;
    demo.game.player.x = ((client_x / width) * 30.0 - 15.0).clamp(-13.0, 13.0);
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

fn create_object_pipeline(
    device: &JsValue,
    color_format: &str,
    depth_format: &str,
    frame_layout: &JsValue,
) -> JsResult<JsValue> {
    let module_desc = object();
    set(&module_desc, "label", "Arcade 3D Objects")?;
    set(&module_desc, "code", OBJECT_SHADER)?;
    let module = call_method(device, "createShaderModule", &[module_desc.into()])?;

    let layouts = Array::new();
    layouts.push(frame_layout);
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

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
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

fn required_element(document: &Document, id: &str) -> JsResult<Element> {
    document
        .get_element_by_id(id)
        .ok_or_else(|| JsValue::from_str(&format!("Missing element #{id}")))
}

fn inject_style(document: &Document) -> JsResult<()> {
    let style = document.create_element("style")?;
    style.set_text_content(Some(
        r#"
        html, body {
          height: 100%;
          margin: 0;
          font-family: Orbitron, ui-monospace, monospace;
          background:
            radial-gradient(circle at 50% 18%, rgba(255, 0, 156, 0.28), transparent 30%),
            linear-gradient(#080016, #000);
          overflow: hidden;
        }

        body::before {
          content: "";
          position: fixed;
          inset: 0;
          z-index: 1;
          pointer-events: none;
          background:
            linear-gradient(rgba(255,255,255,0.035) 50%, rgba(0,0,0,0.08) 50%),
            radial-gradient(ellipse at 50% 95%, rgba(0, 240, 255, 0.2), transparent 35%);
          background-size: 100% 4px, 100% 100%;
          mix-blend-mode: screen;
        }

        canvas {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          touch-action: none;
        }

        .hud {
          position: fixed;
          top: 16px;
          left: 50%;
          transform: translateX(-50%);
          z-index: 3;
          width: min(760px, calc(100vw - 28px));
          display: flex;
          justify-content: space-between;
          gap: 12px;
          padding: 12px 16px;
          box-sizing: border-box;
          color: #fff;
          background: linear-gradient(90deg, rgba(7, 0, 30, 0.72), rgba(32, 0, 62, 0.62));
          border: 1px solid rgba(0, 240, 255, 0.5);
          box-shadow:
            0 0 18px rgba(0, 240, 255, 0.28),
            inset 0 0 18px rgba(255, 0, 204, 0.18);
          clip-path: polygon(14px 0, 100% 0, calc(100% - 14px) 100%, 0 100%);
          text-shadow: 0 0 10px rgba(0, 240, 255, 0.9);
        }

        .hud div {
          display: grid;
          gap: 3px;
          min-width: 92px;
        }

        .hud span {
          font-size: 10px;
          letter-spacing: 0.22em;
          color: #ff75de;
        }

        .hud strong {
          font-size: clamp(16px, 3.2vw, 28px);
          line-height: 1;
        }

        .game-field,
        .ship {
          position: fixed;
          inset: 0;
          z-index: 2;
          pointer-events: none;
        }

        .ship {
          left: calc(var(--x) * 1vw);
          top: calc(var(--y) * 1vh);
          width: 54px;
          height: 70px;
          transform: translate(-50%, -50%) rotate(var(--tilt));
          filter:
            drop-shadow(0 0 10px #00f0ff)
            drop-shadow(0 0 22px rgba(255, 0, 204, 0.85));
          background:
            linear-gradient(135deg, transparent 0 25%, #00f0ff 25% 36%, transparent 36%),
            linear-gradient(225deg, transparent 0 25%, #ff00cc 25% 36%, transparent 36%),
            linear-gradient(#ffffff, #63f8ff 45%, #ff2bd6 46% 62%, transparent 62%);
          clip-path: polygon(50% 0, 96% 84%, 62% 72%, 50% 100%, 38% 72%, 4% 84%);
        }

        .ship::after {
          content: "";
          position: absolute;
          left: 50%;
          top: 74%;
          width: 18px;
          height: 38px;
          transform: translateX(-50%);
          background: linear-gradient(#fff, #fffb00 25%, #ff4b00 70%, transparent);
          clip-path: polygon(50% 100%, 0 0, 100% 0);
          opacity: 0.86;
        }

        .beam,
        .enemy,
        .pickup {
          position: absolute;
          left: calc(var(--x) * 1vw);
          top: calc(var(--y) * 1vh);
          transform: translate(-50%, -50%);
          display: block;
        }

        .beam {
          width: 5px;
          height: 34px;
          border-radius: 999px;
          background: #fff;
          box-shadow: 0 0 12px #fff, 0 0 24px #00f0ff;
        }

        .enemy {
          width: calc(var(--s) * 1.65vw);
          height: calc(var(--s) * 1.65vw);
          min-width: 38px;
          min-height: 38px;
          background:
            linear-gradient(45deg, transparent 0 18%, #ff00cc 18% 35%, transparent 35%),
            linear-gradient(135deg, transparent 0 18%, #00f0ff 18% 35%, transparent 35%),
            radial-gradient(circle, #fff 0 10%, #ff00cc 12% 32%, #280036 34% 100%);
          clip-path: polygon(50% 0, 100% 50%, 50% 100%, 0 50%);
          box-shadow: 0 0 18px rgba(255, 0, 204, 0.9), inset 0 0 16px rgba(0, 240, 255, 0.8);
        }

        .enemy.e1 {
          border-radius: 50%;
          clip-path: polygon(50% 0, 86% 16%, 100% 50%, 86% 84%, 50% 100%, 14% 84%, 0 50%, 14% 16%);
          background:
            radial-gradient(circle, #fff 0 8%, #00f0ff 11% 31%, #11002d 33% 100%);
          box-shadow: 0 0 18px rgba(0, 240, 255, 0.9), inset 0 0 16px rgba(255, 0, 204, 0.7);
        }

        .pickup {
          width: 30px;
          height: 30px;
          background: radial-gradient(circle, #fff 0 13%, #fffb00 16% 43%, transparent 46%);
          box-shadow: 0 0 18px #fffb00, 0 0 34px rgba(255, 0, 204, 0.6);
          clip-path: polygon(50% 0, 62% 36%, 100% 50%, 62% 64%, 50% 100%, 38% 64%, 0 50%, 38% 36%);
        }

        .marquee {
          position: fixed;
          left: 50%;
          top: 50%;
          z-index: 4;
          transform: translate(-50%, -50%) skewX(-8deg);
          padding: 20px 28px;
          max-width: calc(100vw - 40px);
          color: #fff;
          font-size: clamp(26px, 7vw, 72px);
          font-weight: 800;
          letter-spacing: 0.08em;
          white-space: nowrap;
          text-align: center;
          text-shadow:
            0 0 8px #fff,
            0 0 22px #00f0ff,
            0 0 42px #ff00cc;
          background: rgba(8, 0, 22, 0.38);
          border: 1px solid rgba(255, 255, 255, 0.24);
          box-shadow: 0 0 40px rgba(255, 0, 204, 0.35);
          transition: opacity 160ms ease, transform 160ms ease;
        }

        .marquee[data-show="false"] {
          opacity: 0;
          transform: translate(-50%, -48%) skewX(-8deg) scale(0.98);
        }

        .error {
          position: fixed;
          z-index: 5;
          inset: 4rem;
          color: #ffb2b2;
          white-space: pre-wrap;
        }

        @media (max-width: 560px) {
          .hud {
            top: 10px;
            padding: 10px;
          }

          .hud div {
            min-width: 0;
          }

          .hud span {
            font-size: 8px;
          }

          .ship {
            width: 42px;
            height: 56px;
          }
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
