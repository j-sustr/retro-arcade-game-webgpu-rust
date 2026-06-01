//! Infinite spawner for the endless-runner illusion.
//!
//! This is the treadmill system: objects live in world-space track coordinates,
//! then become visible relative to the rider's current progress. When an object
//! falls behind the despawn trigger it is deactivated and returned to its pool.
//! New obstacles, pickups, and scenery are pulled from those pools and placed
//! ahead of the camera. Mountains are scenery objects with stable world
//! positions, so they do not slide with player steering or re-randomize every
//! frame.

#[derive(Clone, Copy)]
pub struct ActiveObject {
    pub x: f32,
    pub z: f32,
    pub size: f32,
    pub kind: u32,
}

#[derive(Clone, Copy)]
pub struct ActiveMountain {
    pub x: f32,
    pub z: f32,
    pub width: f32,
    pub height: f32,
    pub kind: u32,
}

pub struct InfiniteSpawner {
    obstacle_pool: ObjectPool<TrackObject>,
    pickup_pool: ObjectPool<TrackObject>,
    mountain_pool: ObjectPool<MountainObject>,
    next_obstacle_z: f32,
    next_pickup_z: f32,
    next_mountain_z: f32,
    rng: u32,
}

impl InfiniteSpawner {
    const SPAWN_AHEAD: f32 = 240.0;
    const DESPAWN_BEHIND: f32 = 28.0;

    pub fn new() -> Self {
        let mut spawner = Self {
            obstacle_pool: ObjectPool::with_capacity(36),
            pickup_pool: ObjectPool::with_capacity(10),
            mountain_pool: ObjectPool::with_capacity(64),
            next_obstacle_z: 70.0,
            next_pickup_z: 120.0,
            next_mountain_z: 24.0,
            rng: 0x8b17_2084,
        };
        spawner.seed_mountains();
        spawner
    }

    pub fn reset(&mut self, seed: u32) {
        self.obstacle_pool.deactivate_all();
        self.pickup_pool.deactivate_all();
        self.mountain_pool.deactivate_all();
        self.next_obstacle_z = 70.0;
        self.next_pickup_z = 120.0;
        self.next_mountain_z = 24.0;
        self.rng = seed ^ 0xa5a5_2084;
        self.seed_mountains();
    }

    pub fn update(&mut self, progress: f32, wave: u32) {
        self.despawn_passed(progress);
        self.spawn_obstacles(progress, wave);
        self.spawn_pickups(progress);
        self.spawn_mountains(progress);
    }

    pub fn obstacles(&self, progress: f32) -> Vec<ActiveObject> {
        self.obstacle_pool
            .active()
            .map(|object| ActiveObject {
                x: object.x,
                z: relative_z(progress, object.world_z),
                size: object.size,
                kind: object.kind,
            })
            .collect()
    }

    pub fn pickups(&self, progress: f32) -> Vec<ActiveObject> {
        self.pickup_pool
            .active()
            .map(|object| ActiveObject {
                x: object.x,
                z: relative_z(progress, object.world_z),
                size: object.size,
                kind: object.kind,
            })
            .collect()
    }

    pub fn mountains(&self, progress: f32) -> Vec<ActiveMountain> {
        self.mountain_pool
            .active()
            .map(|mountain| ActiveMountain {
                x: mountain.x,
                z: relative_z(progress, mountain.world_z),
                width: mountain.width,
                height: mountain.height,
                kind: mountain.kind,
            })
            .collect()
    }

    pub fn hit_obstacle(&mut self, player_x: f32, progress: f32) -> bool {
        if let Some(hit) = self.obstacle_pool.iter_active_mut().find(|object| {
            let z = relative_z(progress, object.world_z);
            z > -2.4 && z < 4.4 && (player_x - object.x).abs() < object.size * 0.72 + 0.9
        }) {
            hit.active = false;
            true
        } else {
            false
        }
    }

    pub fn collect_pickup(&mut self, player_x: f32, progress: f32) -> bool {
        if let Some(pickup) = self.pickup_pool.iter_active_mut().find(|object| {
            let z = relative_z(progress, object.world_z);
            z > -3.5 && z < 5.0 && (player_x - object.x).abs() < 2.8
        }) {
            pickup.active = false;
            true
        } else {
            false
        }
    }

    fn seed_mountains(&mut self) {
        while self.next_mountain_z < Self::SPAWN_AHEAD {
            self.spawn_mountain_pair();
        }
    }

    fn despawn_passed(&mut self, progress: f32) {
        let cutoff = progress - Self::DESPAWN_BEHIND;
        self.obstacle_pool
            .deactivate_where(|object| object.world_z < cutoff);
        self.pickup_pool
            .deactivate_where(|object| object.world_z < cutoff);
        self.mountain_pool
            .deactivate_where(|mountain| mountain.world_z < cutoff);
    }

    fn spawn_obstacles(&mut self, progress: f32, wave: u32) {
        let spawn_until = progress + Self::SPAWN_AHEAD;
        while self.next_obstacle_z < spawn_until {
            let spacing = (38.0 - wave as f32 * 1.8).max(18.0);
            self.next_obstacle_z += self.random_range(spacing * 0.75, spacing * 1.35);
            let object = TrackObject {
                active: true,
                x: self.random_range(-12.0, 12.0),
                world_z: self.next_obstacle_z,
                size: self.random_range(1.8, 3.8),
                kind: if self.rng & 1 == 0 { 0 } else { 1 },
            };
            self.obstacle_pool.activate(object);
        }
    }

    fn spawn_pickups(&mut self, progress: f32) {
        let spawn_until = progress + Self::SPAWN_AHEAD;
        while self.next_pickup_z < spawn_until {
            self.next_pickup_z += self.random_range(170.0, 260.0);
            let pickup = TrackObject {
                active: true,
                x: self.random_range(-11.0, 11.0),
                world_z: self.next_pickup_z,
                size: 1.8,
                kind: 0,
            };
            self.pickup_pool.activate(pickup);
        }
    }

    fn spawn_mountains(&mut self, progress: f32) {
        let spawn_until = progress + Self::SPAWN_AHEAD;
        while self.next_mountain_z < spawn_until {
            self.spawn_mountain_pair();
        }
    }

    fn spawn_mountain_pair(&mut self) {
        self.next_mountain_z += self.random_range(14.0, 24.0);
        let base_z = self.next_mountain_z;
        let left = MountainObject {
            active: true,
            x: -36.0 - self.random_range(0.0, 24.0),
            world_z: base_z,
            width: self.random_range(10.0, 19.0),
            height: self.random_range(8.0, 26.0),
            kind: 0,
        };
        let right = MountainObject {
            active: true,
            x: 36.0 + self.random_range(0.0, 24.0),
            world_z: base_z + self.random_range(-5.0, 7.0),
            width: self.random_range(9.0, 17.0),
            height: self.random_range(9.0, 29.0),
            kind: 1,
        };
        self.mountain_pool.activate(left);
        self.mountain_pool.activate(right);
    }

    fn random_range(&mut self, min: f32, max: f32) -> f32 {
        self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = ((self.rng >> 8) as f32) / ((u32::MAX >> 8) as f32);
        min + (max - min) * unit
    }
}

fn relative_z(progress: f32, world_z: f32) -> f32 {
    4.0 - (world_z - progress)
}

#[derive(Clone, Copy)]
struct TrackObject {
    active: bool,
    x: f32,
    world_z: f32,
    size: f32,
    kind: u32,
}

#[derive(Clone, Copy)]
struct MountainObject {
    active: bool,
    x: f32,
    world_z: f32,
    width: f32,
    height: f32,
    kind: u32,
}

trait PoolItem {
    fn is_active(&self) -> bool;
    fn set_active(&mut self, active: bool);
}

impl PoolItem for TrackObject {
    fn is_active(&self) -> bool {
        self.active
    }

    fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

impl PoolItem for MountainObject {
    fn is_active(&self) -> bool {
        self.active
    }

    fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

struct ObjectPool<T> {
    objects: Vec<T>,
}

impl<T: PoolItem + Copy> ObjectPool<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            objects: Vec::with_capacity(capacity),
        }
    }

    fn activate(&mut self, object: T) {
        if let Some(slot) = self.objects.iter_mut().find(|item| !item.is_active()) {
            *slot = object;
        } else {
            self.objects.push(object);
        }
    }

    fn active(&self) -> impl Iterator<Item = &T> {
        self.objects.iter().filter(|item| item.is_active())
    }

    fn iter_active_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.objects.iter_mut().filter(|item| item.is_active())
    }

    fn deactivate_all(&mut self) {
        for object in &mut self.objects {
            object.set_active(false);
        }
    }

    fn deactivate_where(&mut self, mut should_deactivate: impl FnMut(&T) -> bool) {
        for object in &mut self.objects {
            if object.is_active() && should_deactivate(object) {
                object.set_active(false);
            }
        }
    }
}
