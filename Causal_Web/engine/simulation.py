# engine/simulation.py

import time
from threading import Thread
from ..config import Config


def simulation_loop():
    def run():
        while True:
            with Config.state_lock:
                running = Config.is_running
                rate = Config.tick_rate
                tick = Config.current_tick
                max_ticks = Config.max_ticks
            if running:
                tick += 1
                with Config.state_lock:
                    Config.current_tick = tick
                print(f"Tick: {tick}")
                if max_ticks and tick >= max_ticks:
                    with Config.state_lock:
                        Config.is_running = False
                time.sleep(rate)
            else:
                time.sleep(0.1)  # Wait until resumed

    Thread(target=run, daemon=True).start()
