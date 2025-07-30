# GUI Usage

The PySide6 dashboard provides an interactive editor for building and running graphs. The **Graph View** window displays nodes and connections and supports standard editing gestures:

- Drag between nodes to create edges or bridges.
- Right-click empty space to add nodes, observers or meta nodes.
- Right-click existing items to delete them.
- Use **Ctrl+Z** and **Ctrl+Y** for undo and redo.
- The toolbar offers quick actions for adding nodes, connections, observers and for triggering auto layout.

When a node or edge is selected a panel appears allowing its parameters to be edited. The panel updates live while dragging and changes are applied with the **Apply** button. Observers and meta nodes use similar panels and remember their targeted nodes when reopened.

The **Control Panel** window lets you start, pause or stop the simulation and set the tick limit and rate. A tick counter shows the current tick. Window resizing keeps the graph responsive and rendering updates only when the graph changes to reduce idle CPU usage.

Saved graphs are copied into each run directory when the simulation starts so the exact input is preserved. The GUI also exposes a **Log Files** window to enable or disable specific logs.
