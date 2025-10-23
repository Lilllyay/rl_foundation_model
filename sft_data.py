import json
import random
from pathlib import Path
import random
import math

PARAMS = {
    # irrigation
    "swfac_irrigate_threshold": 0.92,   # looser threshold = more sensitive
    "swfac_strong_threshold": 0.80,     # stronger trigger
    "max_irrigation_mm": 45.0,
    "min_irrigation_mm": 5.0,
    # fertilizer
    "nstres_threshold": 0.95,           # more sensitive to mild N stress
    "max_fert_per_application_kg_ha": 30.0,
    "seasonal_max_fert_kg_ha": 300.0,
    # scaling by stage
    "stage_irrigation_multiplier": {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.3, 4: 1.1, 5: 0.6},
    "stage_fert_multiplier": {0: 0.2, 1: 0.5, 2: 1.0, 3: 1.1, 4: 0.8, 5: 0.0},
    # fallback triggers
    "topsoil_dry_threshold": 0.22,      # volumetric water content
    "high_et_threshold": 6.0,           # mm/day evapotranspiration
    # randomness
    "allow_small_randomness": True,
    "randomness_fraction": 0.12
}

def _apply_randomness(x, frac):
    return round(x * (1.0 + random.uniform(-frac, frac)), 3)

def mm_to_liters_per_ha(mm):
    return mm * 10000.0  # 1 mm/ha = 10,000 L

def expert_policy(obs, field_area_ha=None):
    """
    Heuristic expert policy for Gym-DSSAT simplified observation space.
    Decides irrigation (mm) and fertilizer (kg/ha) based on soil moisture,
    evapotranspiration, crop stage, rainfall forecast, and fertilizer timing.
    """

    # ---- Extract observation values ----
    soil_moisture = float(obs.get("soil_moisture", 0.25))  # 0–1 scale
    evap = float(obs.get("evapotranspiration", 5.0))        # mm/day
    stage_map = {
    "emergence": 1,
    "vegetative": 2,
    "flowering": 3,
    "reproductive": 3,   # some DSSAT builds use 'reproductive' instead of 'flowering'
    "grain_filling": 4,
    "maturity": 5,
    "harvest": 6
    }

    raw_stage = obs.get("crop_stage", 2)

    if isinstance(raw_stage, str):
        crop_stage = stage_map.get(raw_stage.lower(), 2)  # default to vegetative
    else:
        try:
            crop_stage = int(raw_stage)
        except (TypeError, ValueError):
            crop_stage = 2
    rain_next = float(obs.get("rainfall_next_3days", 0.0))  # mm
    days_since_fert = int(obs.get("days_since_fertilizer", 15))
    temp = float(obs.get("temperature", 25.0))
    srad = float(obs.get("solar_radiation", 15.0))

    # ---- Initialize decisions ----
    irrigation_mm = 0.0
    fertilizer_kg_ha = 0.0
    rationale = []

    # ===================== IRRIGATION POLICY =====================
    # Ideal soil moisture range: 0.30–0.40
    if soil_moisture < 0.25 and rain_next < 5.0:
        # mild stress
        irrigation_mm = 20 + 10 * (0.30 - soil_moisture)
        rationale.append(f"Soil moisture low ({soil_moisture:.2f}), little rain ahead → irrigate {irrigation_mm:.1f} mm.")
    elif soil_moisture < 0.18:
        # severe stress
        irrigation_mm = 35 + 15 * (0.25 - soil_moisture)
        rationale.append(f"Severe drought (soil moisture={soil_moisture:.2f}) → heavy irrigation {irrigation_mm:.1f} mm.")
    elif rain_next > 15:
        rationale.append(f"Heavy rain forecast ({rain_next:.1f} mm) → skip irrigation.")
    else:
        rationale.append("Soil moisture adequate and rain forecast acceptable → no irrigation.")

    # Stage-dependent adjustment (e.g. flowering/grain filling need more)
    if crop_stage in [3, 4]:
        irrigation_mm *= 1.2
        rationale.append(f"Crop at stage {crop_stage} (high demand) → increase irrigation 20%.")
    
    # ===================== FERTILIZER POLICY =====================
    # Simplified: reapply every 20–30 days unless late in season
    if days_since_fert > 25 and crop_stage < 5:
        fertilizer_kg_ha = 40 + random.uniform(-5, 5)
        rationale.append(f"{days_since_fert} days since last fertilizer → apply {fertilizer_kg_ha:.1f} kg/ha.")
    else:
        rationale.append(f"Fertilizer applied recently ({days_since_fert} days) or late stage → skip.")

    # ===================== Small noise for SFT variety =====================
    irrigation_mm *= random.uniform(0.9, 1.1)
    fertilizer_kg_ha *= random.uniform(0.9, 1.1)

    # ===================== Output =====================
    action = {
        "irrigation_mm": round(irrigation_mm, 2),
        "fertilizer_kg_ha": round(fertilizer_kg_ha, 2)
    }

    if field_area_ha:
        action["irrigation_liters_total"] = round(irrigation_mm * 10000 * field_area_ha, 1)

    return {"action": action, "rationale": " ".join(rationale)}

def format_prompt(obs):
    """Generate consistent textual prompt for the LLM."""
    return (
        f"OBS: day={obs['day']}; crop_stage={obs['crop_stage']}; "
        f"soil_moisture={obs['soil_moisture']:.2f}; rainfall_next_3days={obs['rainfall_next_3days']}mm; "
        f"evapotranspiration={obs['evapotranspiration']}mm/day; days_since_fertilizer={obs['days_since_fertilizer']}; "
        f"temperature={obs['temperature']}°C; solar_radiation={obs['solar_radiation']}MJ/m2. "
    )

def collect_sft_data(env, num_episodes=50, max_steps=200, out_file="sft_data_rationale.jsonl"):
    path = Path(out_file)
    with path.open("w") as f:
        for ep in range(num_episodes):
            obs = env.reset()
            for step in range(max_steps):
                # Convert observation
                obs_dict = {
                    "day": obs["day"],
                    "crop_stage": obs["crop_stage"],
                    "soil_moisture": obs["soil_moisture"],
                    "rainfall_next_3days": obs.get("rainfall_next_3days", 0),
                    "evapotranspiration": obs.get("evapotranspiration", 5),
                    "days_since_fertilizer": obs.get("days_since_fertilizer", 999),
                    "temperature": obs.get("temperature", 30),
                    "solar_radiation": obs.get("solar_radiation", 20)
                }

                prompt = format_prompt(obs_dict)
                # obs = env.reset()
                # print("Observation keys:", obs.keys())
                # for _ in range(10):
                #     print({
                #         "swfac": obs["swfac"],
                #         "nstres": obs["nstres"],
                #         "ep": obs["ep"],
                #         "srad": obs["srad"],
                #         "rain": obs.get("rain", None),
                #     })
                #     obs, reward, done, trunc, info = env.step(env.action_space.sample())
                #     if done:
                #         obs = env.reset()
                #         obs, reward, done, info = env.step(action)
                expert_output = expert_policy(obs_dict)
                response = json.dumps(expert_output)
                entry = {"prompt": prompt, "response": response}
                f.write(json.dumps(entry) + "\n")

                obs, reward, done, info = env.step(expert_output["action"])
                if done:
                    break
    print(f"✅ Saved SFT data to {out_file}")
class MockEnv:
    def reset(self):
        return self._random_obs()
    def step(self, action):
        return self._random_obs(), 0, random.random() < 0.05, {}
    def _random_obs(self):
        return {
            "day": random.randint(1, 120),
            "crop_stage": random.choice(["pre-planting", "vegetative", "flowering", "harvest"]),
            "soil_moisture": round(random.uniform(0.05, 0.35), 2),
            "rainfall_next_3days": random.uniform(0, 30),
            "evapotranspiration": random.uniform(3, 7),
            "days_since_fertilizer": random.randint(0, 60),
            "temperature": random.uniform(25, 35),
            "solar_radiation": random.uniform(15, 28)
        }

mock_env = MockEnv()
collect_sft_data(mock_env, num_episodes=100, max_steps=10)
