use crate::error::Result;
use crate::{components, controls, game_state, ui, utils};

use allegro::*;
use allegro_sys::*;
use nalgebra::{Matrix4, Point2};
use rand::prelude::*;

pub struct Menu
{
	subscreens: Vec<ui::SubScreen>,
}

fn to_f32(pos: Point2<i32>) -> Point2<f32>
{
	Point2::new(pos.x as f32, pos.y as f32)
}

impl Menu
{
	pub fn new(state: &mut game_state::GameState) -> Result<Self>
	{
		state.cache_sprite("data/title.cfg")?;
		state.cache_sprite("data/cursor.cfg")?;
		state.paused = false;
		state.sfx.cache_sample("data/ui1.ogg")?;
		state.sfx.cache_sample("data/ui2.ogg")?;

		Ok(Self {
			subscreens: vec![ui::SubScreen::MainMenu(ui::MainMenu::new(
				state.buffer_width,
				state.buffer_height,
			))],
		})
	}

	pub fn input(
		&mut self, event: &Event, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
		match *event
		{
			Event::MouseAxes { x, y, .. } =>
			{
				if state.track_mouse
				{
					let (x, y) = state.transform_mouse(x as f32, y as f32);
					state.mouse_pos = Point2::new(x as i32, y as i32);
				}
			}
			Event::KeyDown {
				keycode: KeyCode::Escape,
				..
			} =>
			{
				if self.subscreens.len() > 1
				{
					state.sfx.play_sound("data/ui2.ogg").unwrap();
					self.subscreens.pop().unwrap();
					return Ok(None);
				}
			}
			_ => (),
		}
		if let Some(action) = self.subscreens.last_mut().unwrap().input(state, event)
		{
			match action
			{
				ui::Action::Forward(subscreen_fn) =>
				{
					self.subscreens.push(subscreen_fn(
						state,
						state.buffer_width,
						state.buffer_height,
					));
				}
				ui::Action::Start => return Ok(Some(game_state::NextScreen::Game)),
				ui::Action::Quit => return Ok(Some(game_state::NextScreen::Quit)),
				ui::Action::Back =>
				{
					self.subscreens.pop().unwrap();
				}
				_ => (),
			}
		}
		Ok(None)
	}

	pub fn draw(&mut self, state: &game_state::GameState) -> Result<()>
	{
		state.core.clear_to_color(Color::from_rgb_f(0., 0., 0.));
		if self.subscreens.len() == 1
		{
			let sprite = "data/title.cfg";
			let sprite = state
				.get_sprite(sprite)
				.expect(&format!("Could not find sprite: {}", sprite));
			sprite.draw(
				Point2::new(state.buffer_width / 2., 0.),
				0,
				Color::from_rgb_f(1., 1., 1.),
				state,
			);
		}
		// This is dumb.
		let sprite = state.get_sprite("data/cursor.cfg").unwrap();
		sprite.draw(
			to_f32(state.mouse_pos),
			0,
			Color::from_rgb_f(1., 1., 1.),
			state,
		);

		self.subscreens.last().unwrap().draw(state);
		Ok(())
	}
}
