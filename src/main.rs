#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]

mod astar;
mod atlas;
mod components;
mod controls;
mod error;
mod game;
mod game_state;
//mod menu;
//mod sfx;
mod sprite;
//mod ui;
mod utils;

use crate::error::Result;
use allegro::*;
use allegro_dialog::*;
use allegro_sys::*;
use rand::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::rc::Rc;
use std::sync;

enum Screen
{
	Game(game::Game),
	//Menu(menu::Menu),
}

fn real_main() -> Result<()>
{
	let mut state = game_state::GameState::new()?;

	let mut flags = OPENGL | PROGRAMMABLE_PIPELINE;

	if state.options.fullscreen
	{
		flags = flags | FULLSCREEN_WINDOW;
	}
	state.core.set_new_display_flags(flags);

	if state.options.vsync_method == 1
	{
		state.core.set_new_display_option(
			DisplayOption::Vsync,
			1,
			DisplayOptionImportance::Suggest,
		);
	}
	let mut display = Display::new(&state.core, state.options.width, state.options.height)
		.map_err(|_| "Couldn't create display".to_string())?;

	let buffer_width = 160;
	let buffer_height = 144;

	let buffer1 = Bitmap::new(&state.core, buffer_width, buffer_height).unwrap();
	let buffer2 = Bitmap::new(&state.core, buffer_width, buffer_height).unwrap();

	state.display_width = display.get_width() as f32;
	state.display_height = display.get_height() as f32;
	state.buffer_width = buffer_width as f32;
	state.buffer_height = buffer_height as f32;
	state.draw_scale = utils::min(
		(display.get_width() as f32) / (buffer_width as f32),
		(display.get_height() as f32) / (buffer_height as f32),
	)
	.floor();

	let timer = Timer::new(&state.core, utils::DT as f64)
		.map_err(|_| "Couldn't create timer".to_string())?;

	let queue =
		EventQueue::new(&state.core).map_err(|_| "Couldn't create event queue".to_string())?;
	queue.register_event_source(display.get_event_source());
	queue.register_event_source(
		state
			.core
			.get_keyboard_event_source()
			.expect("Couldn't get keyboard"),
	);
	queue.register_event_source(
		state
			.core
			.get_mouse_event_source()
			.expect("Couldn't get mouse"),
	);
	queue.register_event_source(timer.get_event_source());

	let mut quit = false;
	let mut draw = true;
	//~ let mut rng = thread_rng();

	let mut cur_screen = Screen::Game(game::Game::new(
		&mut state,
	)?);

	let mut logics_without_draw = 0;
	let mut old_fullscreen = state.options.fullscreen;
	//let mut old_mouse_hide = state.hide_mouse;
	let mut prev_frame_start = state.core.get_time();

	timer.start();
	while !quit
	{
		if draw && queue.is_empty()
		{
			if state.display_width != display.get_width() as f32
				|| state.display_height != display.get_height() as f32
			{
				state.display_width = display.get_width() as f32;
				state.display_height = display.get_height() as f32;
				state.draw_scale = utils::min(
					(display.get_width() as f32) / (buffer_width as f32),
					(display.get_height() as f32) / (buffer_height as f32),
				)
				.floor();
			}

			let frame_start = state.core.get_time();
			state.core.set_target_bitmap(Some(&buffer1));

			match &mut cur_screen
			{
				Screen::Game(game) => game.draw(&state)?,
			}

			if state.options.vsync_method == 2
			{
				state.core.wait_for_vsync().ok();
			}

			state.core.set_target_bitmap(Some(&buffer2));

			//state
			//	.core
			//	.use_shader(Some(&*teleport_shader.upgrade().unwrap()))
			//	.unwrap();
			//state
			//	.core
			//	.set_shader_uniform(
			//		"bitmap_dims",
			//		&[[buffer1.get_width() as f32, buffer1.get_height() as f32]][..],
			//	)
			//	.ok();
			//state
			//	.core
			//	.set_shader_uniform("swirl_amount", &[state.swirl_amount][..])
			//	.ok();

			state.core.draw_bitmap(&buffer1, 0., 0., Flag::zero());

			state.core.set_target_bitmap(Some(display.get_backbuffer()));

			state.core.clear_to_color(Color::from_rgb_f(0., 0., 0.));

			let bw = buffer_width as f32;
			let bh = buffer_height as f32;
			let dw = display.get_width() as f32;
			let dh = display.get_height() as f32;

			state.core.draw_scaled_bitmap(
				&buffer2,
				0.,
				0.,
				bw,
				bh,
				dw / 2. - bw / 2. * state.draw_scale,
				dh / 2. - bh / 2. * state.draw_scale,
				bw * state.draw_scale,
				bh * state.draw_scale,
				Flag::zero(),
			);

			state.core.flip_display();

			if state.tick % 120 == 0
			{
				println!("FPS: {:.2}", 1. / (frame_start - prev_frame_start));
			}
			prev_frame_start = frame_start;
			logics_without_draw = 0;
		}

		let event = queue.wait_for_event();
		let mut next_screen = match &mut cur_screen
		{
			Screen::Game(game) => game.input(&event, &mut state)?,
		};

		match event
		{
			Event::DisplayClose { .. } => quit = true,
			Event::DisplayResize { .. } =>
			{
				display
					.acknowledge_resize()
					.map_err(|_| "Couldn't acknowledge resize".to_string())?;
			}
			Event::TimerTick { .. } =>
			{
				if logics_without_draw > 10
				{
					continue;
				}

				if next_screen.is_none()
				{
					next_screen = match &mut cur_screen
					{
						Screen::Game(game) => game.logic(&mut state)?,
						_ => None,
					}
				}

				if old_fullscreen != state.options.fullscreen
				{
					display.set_flag(FULLSCREEN_WINDOW, state.options.fullscreen);
					old_fullscreen = state.options.fullscreen;
				}

				logics_without_draw += 1;
				//state.sfx.update_sounds()?;

				if !state.paused
				{
					state.tick += 1;
				}
				draw = true;
			}
			_ => (),
		}

		//if let Some(next_screen) = next_screen
		//{
		//	match next_screen
		//	{
		//		game_state::NextScreen::Game {
		//			seed,
		//			restart_music,
		//		} =>
		//		{
		//			cur_screen = CurScreen::Game(map::Map::new(
		//				&mut state,
		//				buffer_width as f32,
		//				buffer_height as f32,
		//				seed,
		//				restart_music,
		//			)?);
		//		}
		//		game_state::NextScreen::Menu =>
		//		{
		//			cur_screen = CurScreen::Menu(menu::Menu::new(
		//				&mut state,
		//				buffer_width as f32,
		//				buffer_height as f32,
		//			)?);
		//		}
		//		game_state::NextScreen::Quit =>
		//		{
		//			quit = true;
		//		}
		//	}
		//}
	}

	Ok(())
}

allegro_main! {
	use std::panic::catch_unwind;

	match catch_unwind(|| real_main().unwrap())
	{
		Err(e) =>
		{
			let err: String = e
				.downcast_ref::<&'static str>()
				.map(|&e| e.to_owned())
				.or_else(|| e.downcast_ref::<String>().map(|e| e.clone()))
				.unwrap_or("Unknown error!".to_owned());

			let mut lines = vec![];
			for line in err.lines().take(10)
			{
				lines.push(line.to_string());
			}
			show_native_message_box(
				None,
				"Error!",
				"An error has occurred!",
				&lines.join("\n"),
				Some("You make me sad."),
				MESSAGEBOX_ERROR,
			);
		}
		Ok(_) => (),
	}
}
