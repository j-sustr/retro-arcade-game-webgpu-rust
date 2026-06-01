use crate::game::GameMode;
use crate::{GridDemo, JsResult};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{KeyboardEvent, PointerEvent};

pub fn attach_input_handlers(demo: &Rc<RefCell<GridDemo>>) -> JsResult<()> {
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
        position_player_from_pointer(&mut demo, event.client_x() as f32);
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
        position_player_from_pointer(&mut demo, event.client_x() as f32);
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
    pub(crate) fn player_tilt_from_pointer(&mut self, x: f32) {
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

fn position_player_from_pointer(demo: &mut GridDemo, client_x: f32) {
    let width = demo.canvas.client_width().max(1) as f32;
    demo.game.player.x = ((client_x / width) * 30.0 - 15.0).clamp(-13.0, 13.0);
}
