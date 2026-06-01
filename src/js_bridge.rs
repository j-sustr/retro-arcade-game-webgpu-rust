use crate::JsResult;
use js_sys::{Array, Function, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

pub async fn await_promise(promise: Promise) -> JsResult<JsValue> {
    JsFuture::from(promise).await
}

pub fn object() -> Object {
    Object::new()
}

pub fn get<T: AsRef<JsValue>>(object: &T, key: &str) -> JsResult<JsValue> {
    Reflect::get(object.as_ref(), &JsValue::from_str(key))
}

pub fn set<T: Into<JsValue>>(object: &Object, key: &str, value: T) -> JsResult<()> {
    Reflect::set(object, &JsValue::from_str(key), &value.into()).map(|_| ())
}

pub fn call_method<T: AsRef<JsValue>>(
    object: &T,
    key: &str,
    args: &[JsValue],
) -> JsResult<JsValue> {
    let function = get(object, key)?.dyn_into::<Function>()?;
    let args_array = Array::new();
    for arg in args {
        args_array.push(arg);
    }
    Reflect::apply(&function, object.as_ref(), &args_array)
}

pub fn show_error(message: &str) {
    if let Some(document) = web_sys::window().and_then(|window| window.document()) {
        if let Ok(error) = document.create_element("pre") {
            error.set_class_name("error");
            error.set_text_content(Some(message));
            let _ = document.body().unwrap().append_child(&error);
        }
    }
    web_sys::console::error_1(&JsValue::from_str(message));
}

pub fn format_js_error(error: JsValue) -> String {
    if let Some(message) = error.as_string() {
        message
    } else {
        format!("{error:?}")
    }
}

pub fn console_error_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
    }));
}
