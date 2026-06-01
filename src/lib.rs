use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect, Uint32Array};
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    Document, Element, HtmlCanvasElement, KeyboardEvent, PointerEvent, WheelEvent, Window,
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
const FIELD_WIDTH: f32 = 100.0;
const FIELD_HEIGHT: f32 = 100.0;

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
    field: Element,
    ship: Element,
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
        let arcade = create_arcade_overlay(&document)?;

        let demo = Rc::new(RefCell::new(Self {
            window,
            document,
            canvas,
            field: arcade.field,
            ship: arcade.ship,
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
            vertex_buffer,
            index_buffer,
            depth_texture: None,
            clear_color: [0.015, 0.0, 0.08, 1.0],
            line_color: [0.0, 0.96, 1.0, 0.92],
            base_color: [0.01, 0.0, 0.035, 1.0],
            line_width: [0.035, 0.035],
            camera: OrbitCamera::new(),
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
        self.render_arcade()?;

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

        Ok(())
    }

    fn write_grid_uniforms(&self, buffer: &JsValue) -> JsResult<()> {
        let mut uniforms = [0.0; 16];
        uniforms[0..4].copy_from_slice(&self.line_color);
        uniforms[4..8].copy_from_slice(&self.base_color);
        uniforms[8..10].copy_from_slice(&self.line_width);
        write_f32_buffer(&self.queue, buffer, &uniforms)
    }

    fn render_arcade(&self) -> JsResult<()> {
        self.ship.set_attribute(
            "style",
            &format!(
                "--x:{:.3}; --y:{:.3}; --tilt:{:.3}deg;",
                self.game.player.x, self.game.player.y, self.game.player.tilt
            ),
        )?;

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

        let mut html = String::new();
        for shot in &self.game.shots {
            html.push_str(&format!(
                r#"<i class="beam" style="--x:{:.3};--y:{:.3};"></i>"#,
                shot.x, shot.y
            ));
        }
        for enemy in &self.game.enemies {
            html.push_str(&format!(
                r#"<i class="enemy e{}" style="--x:{:.3};--y:{:.3};--s:{:.3};"></i>"#,
                enemy.kind, enemy.x, enemy.y, enemy.size
            ));
        }
        for pickup in &self.game.pickups {
            html.push_str(&format!(
                r#"<i class="pickup" style="--x:{:.3};--y:{:.3};"></i>"#,
                pickup.x, pickup.y
            ));
        }
        self.field.set_inner_html(&html);
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
    field: Element,
    ship: Element,
    hud_score: Element,
    hud_shields: Element,
    hud_wave: Element,
    marquee: Element,
}

struct GameState {
    mode: GameMode,
    paused: bool,
    player: Player,
    enemies: Vec<Entity>,
    shots: Vec<Entity>,
    pickups: Vec<Entity>,
    input: InputState,
    score: u32,
    shields: u32,
    wave: u32,
    enemy_timer: f32,
    pickup_timer: f32,
    fire_cooldown: f32,
    rng: u32,
}

struct Player {
    x: f32,
    y: f32,
    tilt: f32,
}

#[derive(Clone)]
struct Entity {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    size: f32,
    kind: u32,
}

#[derive(Default)]
struct InputState {
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fire: bool,
}

impl GameState {
    fn new() -> Self {
        Self {
            mode: GameMode::Attract,
            paused: false,
            player: Player {
                x: 50.0,
                y: 83.0,
                tilt: 0.0,
            },
            enemies: Vec::new(),
            shots: Vec::new(),
            pickups: Vec::new(),
            input: InputState::default(),
            score: 0,
            shields: 3,
            wave: 1,
            enemy_timer: 0.2,
            pickup_timer: 4.0,
            fire_cooldown: 0.0,
            rng: 0x2084_1986,
        }
    }

    fn start(&mut self) {
        let rng = self.rng.wrapping_add(0x9e37_79b9);
        *self = Self::new();
        self.rng = rng;
        self.mode = GameMode::Playing;
    }

    fn update(&mut self, dt: f32) {
        if self.mode != GameMode::Playing || self.paused {
            return;
        }

        let mut dx: f32 = 0.0;
        let mut dy: f32 = 0.0;
        if self.input.left {
            dx -= 1.0;
        }
        if self.input.right {
            dx += 1.0;
        }
        if self.input.up {
            dy -= 1.0;
        }
        if self.input.down {
            dy += 1.0;
        }

        let speed = 46.0 + self.wave as f32 * 2.0;
        self.player.x = (self.player.x + dx * speed * dt).clamp(8.0, 92.0);
        self.player.y = (self.player.y + dy * speed * dt).clamp(58.0, 91.0);
        self.player.tilt += ((dx * 18.0) - self.player.tilt) * (dt * 12.0).min(1.0);

        self.fire_cooldown = (self.fire_cooldown - dt).max(0.0);
        if self.input.fire && self.fire_cooldown <= 0.0 {
            self.fire_cooldown = 0.16;
            self.shots.push(Entity {
                x: self.player.x,
                y: self.player.y - 6.0,
                vx: 0.0,
                vy: -90.0,
                size: 2.6,
                kind: 0,
            });
        }

        self.enemy_timer -= dt;
        if self.enemy_timer <= 0.0 {
            let wave_speed = 18.0 + self.wave as f32 * 3.2;
            self.enemy_timer = (0.85 - self.wave as f32 * 0.045).max(0.26);
            let lane = self.random_range(10.0, 90.0);
            let drift = self.random_range(-10.0, 10.0);
            let speed = wave_speed + self.random_range(0.0, 12.0);
            let size = self.random_range(5.0, 8.5);
            let kind = if self.rng & 1 == 0 { 0 } else { 1 };
            self.enemies.push(Entity {
                x: lane,
                y: -8.0,
                vx: drift,
                vy: speed,
                size,
                kind,
            });
        }

        self.pickup_timer -= dt;
        if self.pickup_timer <= 0.0 {
            self.pickup_timer = self.random_range(5.5, 8.5);
            let x = self.random_range(12.0, 88.0);
            self.pickups.push(Entity {
                x,
                y: -8.0,
                vx: 0.0,
                vy: 24.0,
                size: 4.0,
                kind: 0,
            });
        }

        for shot in &mut self.shots {
            shot.y += shot.vy * dt;
        }
        for enemy in &mut self.enemies {
            enemy.x = (enemy.x + enemy.vx * dt).clamp(5.0, 95.0);
            enemy.y += enemy.vy * dt;
        }
        for pickup in &mut self.pickups {
            pickup.y += pickup.vy * dt;
        }

        self.resolve_hits();
        self.shots.retain(|shot| shot.y > -8.0);
        self.enemies.retain(|enemy| enemy.y < 112.0);
        self.pickups.retain(|pickup| pickup.y < 112.0);

        self.score = self.score.saturating_add((dt * 18.0) as u32);
        self.wave = 1 + self.score / 1200;
    }

    fn resolve_hits(&mut self) {
        let mut dead_enemies = vec![false; self.enemies.len()];
        let mut dead_shots = vec![false; self.shots.len()];

        for (shot_index, shot) in self.shots.iter().enumerate() {
            for (enemy_index, enemy) in self.enemies.iter().enumerate() {
                if dead_enemies[enemy_index] {
                    continue;
                }
                if distance2(shot.x, shot.y, enemy.x, enemy.y) < (enemy.size + shot.size).powi(2) {
                    dead_enemies[enemy_index] = true;
                    dead_shots[shot_index] = true;
                    self.score = self.score.saturating_add(120 + self.wave * 10);
                    break;
                }
            }
        }

        let px = self.player.x;
        let py = self.player.y;
        for (enemy_index, enemy) in self.enemies.iter().enumerate() {
            if dead_enemies[enemy_index] {
                continue;
            }
            if distance2(px, py, enemy.x, enemy.y) < (enemy.size + 5.5).powi(2) {
                dead_enemies[enemy_index] = true;
                self.shields = self.shields.saturating_sub(1);
                if self.shields == 0 {
                    self.mode = GameMode::GameOver;
                }
            }
        }

        let mut dead_pickups = vec![false; self.pickups.len()];
        for (pickup_index, pickup) in self.pickups.iter().enumerate() {
            if distance2(px, py, pickup.x, pickup.y) < (pickup.size + 6.0).powi(2) {
                dead_pickups[pickup_index] = true;
                self.score = self.score.saturating_add(300);
                self.shields = (self.shields + 1).min(5);
            }
        }

        let mut enemy_index = 0;
        self.enemies.retain(|_| {
            let keep = !dead_enemies[enemy_index];
            enemy_index += 1;
            keep
        });
        let mut shot_index = 0;
        self.shots.retain(|_| {
            let keep = !dead_shots[shot_index];
            shot_index += 1;
            keep
        });
        let mut pickup_index = 0;
        self.pickups.retain(|_| {
            let keep = !dead_pickups[pickup_index];
            pickup_index += 1;
            keep
        });
    }

    fn random_range(&mut self, min: f32, max: f32) -> f32 {
        self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = ((self.rng >> 8) as f32) / ((u32::MAX >> 8) as f32);
        min + (max - min) * unit
    }
}

fn distance2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
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

    let field = document.create_element("div")?;
    field.set_class_name("game-field");
    document.body().unwrap().append_child(&field)?;

    let ship = document.create_element("div")?;
    ship.set_class_name("ship");
    ship.set_attribute("style", "--x:50; --y:83; --tilt:0deg;")?;
    document.body().unwrap().append_child(&ship)?;

    let marquee = document.create_element("div")?;
    marquee.set_class_name("marquee");
    marquee.set_attribute("data-show", "true")?;
    marquee.set_text_content(Some("NEON GRID 2084  START"));
    document.body().unwrap().append_child(&marquee)?;

    Ok(ArcadeOverlay {
        field,
        ship,
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
        demo.game.input.fire = true;
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
            demo.game.input.fire = false;
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
        "ArrowUp" | "w" | "W" => demo.game.input.up = pressed,
        "ArrowDown" | "s" | "S" => demo.game.input.down = pressed,
        " " | "Spacebar" => {
            if pressed {
                if demo.game.mode == GameMode::Playing {
                    demo.game.input.fire = true;
                } else {
                    demo.game.start();
                }
            } else {
                demo.game.input.fire = false;
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

fn position_player_from_pointer(demo: &mut GridDemo, client_x: f32, client_y: f32) {
    let width = demo.canvas.client_width().max(1) as f32;
    let height = demo.canvas.client_height().max(1) as f32;
    demo.game.player.x = (client_x / width * FIELD_WIDTH).clamp(8.0, 92.0);
    demo.game.player.y = (client_y / height * FIELD_HEIGHT).clamp(58.0, 91.0);
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
