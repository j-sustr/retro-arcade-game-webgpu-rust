use crate::infinite_spawner::InfiniteSpawner;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    Attract,
    Playing,
    GameOver,
}

pub struct GameState {
    pub mode: GameMode,
    pub paused: bool,
    pub player: Player,
    pub spawner: InfiniteSpawner,
    pub input: InputState,
    pub score: u32,
    pub shields: u32,
    pub wave: u32,
    pub scroll: f32,
    pub speed: f32,
    rng: u32,
}

pub struct Player {
    pub x: f32,
    pub tilt: f32,
}

#[derive(Default)]
pub struct InputState {
    pub left: bool,
    pub right: bool,
    pub boost: bool,
}

impl GameState {
    pub fn new() -> Self {
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

    pub fn start(&mut self) {
        let rng = self.rng.wrapping_add(0x9e37_79b9);
        *self = Self::new();
        self.rng = rng;
        self.spawner.reset(rng);
        self.mode = GameMode::Playing;
    }

    pub fn update(&mut self, dt: f32) {
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
