use crate::game::GameState;
use crate::infinite_spawner::{ActiveMountain, ActiveObject};

pub const OBJECT_VERTEX_CAPACITY: usize = 24_000;

pub fn build_scene_vertices(game: &GameState) -> Vec<f32> {
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
