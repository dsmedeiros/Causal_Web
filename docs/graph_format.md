# Graph Format

Graphs are described using JSON files with top level keys for nodes, edges, optional bridges, tick_sources, observers and meta_nodes. Each node defines its position and simulation parameters. Edges specify delays, attenuation and optional unitary or gauge terms. Tick sources seed periodic activity while observers record metrics. Meta nodes group related nodes under additional constraints such as phase locking.
Nodes can optionally set ``allow_self_connection`` to ``true`` to permit self-connected edges and ``cnot_source`` to spawn dynamic ε-pairs when they fire.

## Example
```json
{
  "nodes": {
    "A": {
      "x": 100,
      "y": 100,
      "frequency": 1.0,
      "refractory_period": 0.5,
      "base_threshold": 0.1,
      "phase": 0.0,
      "origin_type": "seed",
      "generation_tick": 0,
      "parent_ids": [],
      "goals": {},
      "allow_self_connection": false,
      "cnot_source": false
    },
    "B": {
      "x": 100,
      "y": 200,
      "frequency": 1.0,
      "refractory_period": 0.5,
      "base_threshold": 0.1,
      "phase": 0.0,
      "origin_type": "derived",
      "generation_tick": 0,
      "parent_ids": ["A"]
    }
  },
  "edges": [
    {
      "from": "A",
      "to": "B",
      "attenuation": 1.0,
      "density": 0.0,
      "delay": 1,
      "phase_shift": 0.0,
      "A_phase": 0.0,
      "u_id": 0
    }
  ],
  "bridges": [],
  "tick_sources": [
    {
      "node_id": "A",
      "tick_interval": 2,
      "phase": 0.0
    }
  ],
  "observers": [
    {
      "id": "OBS",
      "monitors": ["collapse", "law_wave", "region"],
      "frequency": 1
    }
  ],
  "meta_nodes": {
    "MN1": {
      "members": ["A", "B"],
      "constraints": {"phase_lock": {"tolerance": 0.1}},
      "type": "Configured",
      "collapsed": false,
      "x": 0.0,
      "y": 0.0
    }
  }
}
```

To define an ε-pair directly, mark both directions with matching ``epsilon`` and ``partner_id`` values:

```json
"edges": [
  {"from": "A", "to": "B", "epsilon": true, "partner_id": "pair1"},
  {"from": "B", "to": "A", "epsilon": true, "partner_id": "pair1"}
]
```

Observers may include optional `x` and `y` fields storing their position on the canvas. Set `detector_mode` to `true` to perform binary measurements on ticks that arrive via entangled bridges. Measurement angles can be overridden with `measurement_settings`, a list of float values. Emergent meta nodes discovered at runtime are logged for analysis but do not affect behaviour unless declared in the file.

Bridges can optionally enable entanglement using `"is_entangled": true`. When
set, an `entangled_id` is generated and saved alongside the bridge metadata.
