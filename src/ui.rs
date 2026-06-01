use crate::JsResult;
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element};

pub struct ArcadeOverlay {
    pub hud_score: Element,
    pub hud_shields: Element,
    pub hud_wave: Element,
    pub marquee: Element,
}

pub fn create_arcade_overlay(document: &Document) -> JsResult<ArcadeOverlay> {
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

pub fn inject_style(document: &Document) -> JsResult<()> {
    let style = document.create_element("style")?;
    style.set_text_content(Some(
        r#"
        html, body {
          height: 100%;
          margin: 0;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
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
        }
        "#,
    ));
    document
        .query_selector("head")?
        .ok_or_else(|| JsValue::from_str("Missing document head"))?
        .append_child(&style)?;
    Ok(())
}

fn required_element(document: &Document, id: &str) -> JsResult<Element> {
    document
        .get_element_by_id(id)
        .ok_or_else(|| JsValue::from_str(&format!("Missing element #{id}")))
}
