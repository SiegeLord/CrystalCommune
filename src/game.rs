use crate::error::Result;
use crate::{game_state, sprite};
use allegro::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point2, Point3, Quaternion, RealField, Rotation2, Rotation3,
	Unit, Vector2, Vector3, Vector4,
};
use nalgebra as na;
use rand::prelude::*;

static TILE_WIDTH: usize = 16;

pub struct Game
{
	map: Map,
}

impl Game
{
	pub fn new(state: &mut game_state::GameState) -> Result<Self>
	{
		Ok(Self {
			map: Map::new(state)?,
		})
	}

	pub fn logic(
		&mut self, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
		self.map.logic(state)
	}

	pub fn input(
		&mut self, event: &Event, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
		self.map.input(event, state)
	}

	pub fn draw(&mut self, state: &game_state::GameState) -> Result<()>
	{
		state.core.clear_to_color(Color::from_rgb_f(0.5, 0.5, 1.));
		self.map.draw(state)?;
		Ok(())
	}
}

struct Map
{
	size: usize,
	terrain: Vec<bool>,
}

#[rustfmt::skip]
fn decode_tile<T: Rng>(vals: [i32; 4], rng: &mut T) -> i32
{
	match vals
    {
        [1, 1,
         1, 1] => rng.gen_range(1..=2),
        
        [0, 0,
         1, 1] => 4,
        [1, 0,
         1, 0] => 5,
        [1, 1,
         0, 0] => 6,
        [0, 1,
         0, 1] => 7,

        [0, 0,
         0, 1] => 8,
        [0, 0,
         1, 0] => 9,
        [1, 0,
         0, 0] => 10,
        [0, 1,
         0, 0] => 11,

        [1, 0,
         0, 1] => 12,
        [0, 1,
         1, 0] => 13,

        [1, 1,
         1, 0] => 14,
        [1, 1,
         0, 1] => 15,
        [1, 0,
         1, 1] => 16,
        [0, 1,
         1, 1] => 17,
        _ => 0,
    }
}

fn xy_to_idx(x: usize, y: usize, size: usize) -> usize
{
	y * size + x
}

impl Map
{
	pub fn new(state: &mut game_state::GameState) -> Result<Self>
	{
		let size = 16;
		let mut terrain = Vec::with_capacity(size * size);
		let mut rng = StdRng::seed_from_u64(0);
		state.cache_sprite("data/terrain.cfg")?;
		for y in 0..size
		{
			for x in 0..size
			{
				let val = if x == 0 || x == size - 1 || y == 0 || y == size - 1
				{
					false
				}
				else
				{
					rng.gen_bool(0.8)
				};
				terrain.push(val)
			}
		}

		Ok(Self {
			size: size,
			terrain: terrain,
		})
	}

	fn logic(&mut self, state: &mut game_state::GameState)
		-> Result<Option<game_state::NextScreen>>
	{
		Ok(None)
	}

	fn input(
		&mut self, event: &Event, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
		Ok(None)
	}

	fn draw(&mut self, state: &game_state::GameState) -> Result<()>
	{
		state.core.clear_to_color(Color::from_rgb_f(0.05, 0.05, 0.1));
		let terrain_sprite = state.get_sprite("data/terrain.cfg").unwrap();

        let mut rng = StdRng::seed_from_u64(0);
		state.core.hold_bitmap_drawing(true);
		for y in 0..self.size - 1
		{
			for x in 0..self.size - 1
			{
				let t1 = self.terrain[xy_to_idx(x, y, self.size)] as i32;
				let t2 = self.terrain[xy_to_idx(x + 1, y, self.size)] as i32;
				let t3 = self.terrain[xy_to_idx(x, y + 1, self.size)] as i32;
				let t4 = self.terrain[xy_to_idx(x + 1, y + 1, self.size)] as i32;

				let tile = decode_tile([t1, t2, t3, t4], &mut rng);
				terrain_sprite.draw(
					Point2::new((x * TILE_WIDTH) as f32, (y * TILE_WIDTH) as f32),
					tile,
					Color::from_rgb_f(1., 1., 1.),
					state,
				);
			}
		}
		state.core.hold_bitmap_drawing(false);
		Ok(())
	}
}
