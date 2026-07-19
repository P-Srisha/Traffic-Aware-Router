"""
astar_router.py  —  India Multi-City A* Traffic Router
=======================================================
Loads pre-generated city CSVs from india_road_data/ folder.
User selects State → City → Source node → Dest node → Run A*.
Live step-by-step animation in Jupyter via clear_output loop.

Folder structure expected:
    india_road_data/
        manifest.json
        Karnataka/
            Bengaluru.csv  Mysuru.csv  Udupi.csv  Mangaluru.csv
        Maharashtra/
            Mumbai.csv  Pune.csv  Nagpur.csv
        ...  (13 more states)
"""

import os, json, heapq, math, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

warnings.filterwarnings("ignore")

HW_LWIDTH = {
    "motorway":4.5,"trunk":3.5,"primary":2.6,
    "secondary":1.7,"tertiary":1.1,"residential":0.7,"unclassified":0.5,
}

# ─────────────────────────────────────────────────────────────────
class RoadNetwork:
    """
    Loads a city CSV from india_road_data/<State>/<City>.csv
    and builds a directed graph for A*.
    """

    def __init__(self, state, city, data_dir="india_road_data"):
        self.state    = state
        self.city     = city
        self.data_dir = data_dir
        self.nodes    = {}   # node_id -> {lat, lon, x, y, name, highway}
        self.edges    = []   # list of edge dicts
        self.adj      = {}   # node_id -> [{to, cost, edge}]
        self.df       = None
        self._load()

    def _load(self):
        state_dir = self.state.replace(" ", "_")
        fpath = os.path.join(self.data_dir, state_dir, f"{self.city}.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"CSV not found: {fpath}")

        df = pd.read_csv(fpath)
        self.df = df

        # ── Build unique nodes from u/v + lat/lon columns ────────
        for _, row in df.iterrows():
            for side in [("u","u_lat","u_lon","u_name"),
                         ("v","v_lat","v_lon","v_name")]:
                nid = int(row[side[0]])
                if nid not in self.nodes:
                    self.nodes[nid] = {
                        "id":   nid,
                        "lat":  float(row[side[1]]),
                        "lon":  float(row[side[2]]),
                        "name": str(row[side[3]]) if side[3] in row.index else str(nid),
                        "highway": str(row.get("highway","unknown")),
                    }

        # Normalise coords to [0,1] for plot
        lats = [n["lat"] for n in self.nodes.values()]
        lons = [n["lon"] for n in self.nodes.values()]
        lat0,lat1 = min(lats),max(lats)
        lon0,lon1 = min(lons),max(lons)
        lr = max(lat1-lat0,1e-6)
        lo = max(lon1-lon0,1e-6)
        for n in self.nodes.values():
            n["x"] = (n["lon"]-lon0)/lo
            n["y"] = (n["lat"]-lat0)/lr

        # ── Build edges ──────────────────────────────────────────
        for _, row in df.iterrows():
            e = {
                "u":               int(row["u"]),
                "v":               int(row["v"]),
                "name":            str(row.get("name","")),
                "highway":         str(row.get("highway","unknown")),
                "length":          float(row["length"]),
                "speed_kph":       float(row["speed_kph"]),
                "travel_time":     float(row["travel_time"]),
                "traffic_volume":  float(row.get("traffic_volume",0)),
                "PCU":             float(row.get("PCU",0)),
                "capacity":        float(row.get("capacity",1800)),
                "vc_ratio":        float(row.get("vc_ratio",0)),
                "congested_time":  float(row["congested_time"]),
                "congestion_class":str(row.get("congestion_class","free")),
            }
            self.edges.append(e)

        # ── Adjacency list ───────────────────────────────────────
        for nid in self.nodes:
            self.adj[nid] = []
        for e in self.edges:
            if e["u"] in self.nodes and e["v"] in self.nodes:
                self.adj[e["u"]].append({
                    "to":   e["v"],
                    "cost": e["congested_time"],
                    "edge": e,
                })

    def heuristic(self, a, b):
        """Haversine distance → seconds at 50 kph."""
        na, nb = self.nodes[a], self.nodes[b]
        R  = 6371000
        la1, la2 = math.radians(na["lat"]), math.radians(nb["lat"])
        dl  = la2 - la1
        dlo = math.radians(nb["lon"] - na["lon"])
        h   = math.sin(dl/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
        return 2*R*math.asin(math.sqrt(h)) / (50/3.6)

    def node_labels(self):
        """Returns list of (display_label, node_id) for dropdowns."""
        out = []
        for nid, n in self.nodes.items():
            lbl = f"{n['name']}  ({n['lat']:.4f}, {n['lon']:.4f})"
            out.append((lbl, nid))
        return out

    def summary(self):
        print("="*56)
        print(f"  {self.city}, {self.state}")
        print(f"  Nodes   : {len(self.nodes)}")
        print(f"  Edges   : {len(self.df)}")
        print(f"  Columns : {list(self.df.columns)}")
        print("="*56)
        return self.df[["u_name","v_name","name","highway",
                         "length","speed_kph","travel_time",
                         "vc_ratio","congested_time",
                         "congestion_class"]].head(8)


# ─────────────────────────────────────────────────────────────────
class AStarEngine:
    """Runs A* and records every step for animation replay."""

    def __init__(self, network, src, dst):
        self.net   = network
        self.src   = src
        self.dst   = dst
        self.steps = []
        self.path  = []
        self.cost  = None
        self._run()

    def _run(self):
        net = self.net
        INF = float("inf")
        g    = {n: INF for n in net.nodes}
        f    = {n: INF for n in net.nodes}
        came = {}
        g[self.src] = 0
        f[self.src] = net.heuristic(self.src, self.dst)

        heap     = [(f[self.src], self.src)]
        open_set = {self.src}
        closed   = set()

        def snap(curr, ev, msg):
            self.steps.append({
                "open":    set(open_set),
                "closed":  set(closed),
                "current": curr,
                "event":   ev,
                "msg":     msg,
            })

        sn = net.nodes[self.src]
        dn = net.nodes[self.dst]
        snap(self.src, "start",
             f"START  {sn['name']}  ->  {dn['name']}")

        while heap:
            _, curr = heapq.heappop(heap)
            if curr not in open_set:
                continue
            open_set.discard(curr)

            if curr == self.dst:
                snap(curr, "found",
                     f"GOAL REACHED  {net.nodes[curr]['name']}")
                self.cost = g[self.dst]
                nd = self.dst
                while nd in came:
                    self.path.insert(0, nd)
                    nd = came[nd]
                self.path.insert(0, self.src)
                return

            closed.add(curr)
            snap(curr, "visit",
                 f"VISIT  {net.nodes[curr]['name']}  "
                 f"g={g[curr]:.0f}s")

            for nb in net.adj[curr]:
                nid = nb["to"]
                if nid in closed:
                    continue
                tg = g[curr] + nb["cost"]
                if tg < g[nid]:
                    came[nid] = curr
                    g[nid]    = tg
                    f[nid]    = tg + net.heuristic(nid, self.dst)
                    heapq.heappush(heap, (f[nid], nid))
                    open_set.add(nid)
                    snap(curr, "explore",
                         f"  OPEN  {net.nodes[nid]['name']}  "
                         f"edge={nb['cost']:.0f}s  g={tg:.0f}s")

        snap(None, "no_path",
             "NO PATH — nodes may not be connected in this CSV")


# ─────────────────────────────────────────────────────────────────
class AStarVisualizer:
    """
    Jupyter interactive A* visualizer.

    STEP 1: Select State   → city dropdown updates automatically
    STEP 2: Select City    → loads its CSV, populates node dropdowns
    STEP 3: Pick src/dst   → click Run A* Live
    """

    C = {
        "bg":"#0d1117","panel":"#161b22","border":"#30363d",
        "node_idle":"#21262d","open":"#0d419d","closed":"#1a3a5c",
        "current":"#d29922","path":"#238636","src":"#1f6feb","dst":"#da3633",
        "text":"#e6edf3","muted":"#484f58",
        "edge_base":"#1c2128","edge_open":"#1f6feb","edge_vis":"#264f78",
        "edge_path":"#3fb950","edge_curr":"#d29922",
    }
    EC = {"start":"#58a6ff","visit":"#d29922","explore":"#6e7681",
          "found":"#3fb950","no_path":"#da3633"}
    EP = {"start":"▶","visit":"◈","explore":"  ·","found":"✓","no_path":"✗"}

    def __init__(self, data_dir="india_road_data"):
        self.data_dir = data_dir
        self.net      = None
        self.manifest = self._load_manifest()
        self._build_ui()

    def _load_manifest(self):
        mpath = os.path.join(self.data_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise FileNotFoundError(
                f"manifest.json not found in {self.data_dir}.\n"
                f"Run generate_city_csvs.py first."
            )
        with open(mpath) as f:
            return json.load(f)

    def _build_ui(self):
        sl   = {"description_width":"130px"}
        w440 = widgets.Layout(width="440px")
        w200 = widgets.Layout(width="200px", height="36px")

        states = sorted(self.manifest.keys())

        self.w_state = widgets.Dropdown(
            options=states, value=states[0],
            description="State:", style=sl, layout=w440)

        self.w_city = widgets.Dropdown(
            options=self.manifest[states[0]],
            value=self.manifest[states[0]][0],
            description="City:", style=sl, layout=w440)

        self.w_load = widgets.Button(
            description="⬇  Load City Network",
            button_style="primary",
            layout=widgets.Layout(width="220px", height="36px"))

        self.w_load_out = widgets.Output()

        self.w_src = widgets.Dropdown(
            options=[], description="Source:", style=sl, layout=w440)
        self.w_dst = widgets.Dropdown(
            options=[], description="Dest:", style=sl, layout=w440)

        self.w_delay = widgets.FloatSlider(
            value=0.18, min=0.02, max=1.2, step=0.02,
            description="Delay (s/step):",
            style={"description_width":"130px"},
            layout=w440, readout_format=".2f")

        self.w_run    = widgets.Button(description="▶  Run A* Live",  button_style="success", layout=w200)
        self.w_static = widgets.Button(description="📊  Final Path",    button_style="info",    layout=w200)
        self.w_table  = widgets.Button(description="📋  Edge Table",    button_style="warning", layout=w200)
        self.w_out    = widgets.Output()

        # Wire up state change → auto-update city list
        self.w_state.observe(self._on_state_change, names="value")
        self.w_load.on_click(self._on_load)
        self.w_run.on_click(self._on_run)
        self.w_static.on_click(self._on_static)
        self.w_table.on_click(self._on_table)

        header = widgets.HTML("""
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                    padding:12px 18px;font-family:monospace;margin-bottom:4px;">
          <span style="color:#58a6ff;font-size:16px;font-weight:700;">
            A* India Traffic Router &mdash; State &rarr; City &rarr; Route
          </span><br>
          <span style="color:#8b949e;font-size:11px;">
            23 cities &nbsp;&middot;&nbsp; 15 states &nbsp;&middot;&nbsp;
            Pre-generated CSVs &nbsp;&middot;&nbsp; BPR congested_time &nbsp;&middot;&nbsp;
            Live A* animation
          </span>
        </div>""")

        sep = widgets.HTML('<hr style="border-color:#30363d;margin:5px 0;">')

        lbl1 = widgets.HTML('<b style="font-family:monospace;color:#8b949e;font-size:11px;">STEP 1 — Select state and city, then load</b>')
        lbl2 = widgets.HTML('<b style="font-family:monospace;color:#8b949e;font-size:11px;">STEP 2 — Pick source and destination, then run</b>')

        self.ui = widgets.VBox([
            header, lbl1,
            widgets.HBox([self.w_state, self.w_city]),
            self.w_load, self.w_load_out,
            sep, lbl2,
            self.w_src, self.w_dst, self.w_delay,
            widgets.HBox([self.w_run, self.w_static, self.w_table],
                         layout=widgets.Layout(gap="8px")),
            self.w_out,
        ], layout=widgets.Layout(gap="5px"))

    def show(self):
        display(self.ui)

    def _on_state_change(self, change):
        state = change["new"]
        cities = self.manifest.get(state, [])
        self.w_city.options = cities
        if cities:
            self.w_city.value = cities[0]

    def _on_load(self, _):
        state = self.w_state.value
        city  = self.w_city.value
        with self.w_load_out:
            clear_output(wait=True)
            print(f"  Loading: {city}, {state} ...")
            try:
                self.net = RoadNetwork(state, city, self.data_dir)
                labels   = self.net.node_labels()
                self.w_src.options = labels
                self.w_dst.options = labels
                all_ids = list(self.net.nodes.keys())
                self.w_src.value = labels[0][1]
                self.w_dst.value = labels[len(all_ids)//2][1]
                print(f"  Ready: {len(self.net.nodes)} nodes, "
                      f"{len(self.net.df)} edge rows")
                print(f"  Pick source & destination then click ▶ Run A* Live")
            except Exception as ex:
                print(f"  ERROR: {ex}")
                import traceback; traceback.print_exc()

    # ── Coordinate helper ─────────────────────────────────────────
    def _xy(self, nid):
        n = self.net.nodes[nid]
        return n["x"], n["y"]

    # ── Core frame draw ───────────────────────────────────────────
    def _draw_frame(self, ax_g, ax_l,
                    open_set, closed, current, src, dst,
                    log_lines, step_num, total,
                    path_nodes=None, path_edge_set=None,
                    is_final=False, final_cost=0, path_len=0):

        C   = self.C
        net = self.net
        path_nodes    = path_nodes    or set()
        path_edge_set = path_edge_set or set()

        ax_g.clear(); ax_l.clear()
        for ax in (ax_g, ax_l):
            ax.set_facecolor(C["bg"])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(C["border"])
        ax_g.set_xlim(-0.04, 1.04)
        ax_g.set_ylim(-0.04, 1.12)

        # ── Edges ────────────────────────────────────────────────
        drawn = set()
        for e in net.edges:
            ek = (min(e["u"],e["v"]), max(e["u"],e["v"]))
            if ek in drawn: continue
            drawn.add(ek)
            if e["u"] not in net.nodes or e["v"] not in net.nodes: continue
            x0,y0 = self._xy(e["u"]); x1,y1 = self._xy(e["v"])
            blw   = HW_LWIDTH.get(e["highway"], 0.8)
            ip    = ek in path_edge_set
            bv    = e["u"] in closed and e["v"] in closed
            ce    = (e["u"]==current or e["v"]==current)
            io    = e["u"] in open_set or e["v"] in open_set

            if ip:                     col,lw,al = C["edge_path"],blw*3.5,1.0
            elif ce and not is_final:  col,lw,al = C["edge_curr"],blw*2.2,0.95
            elif bv:                   col,lw,al = C["edge_vis"], blw*1.5,0.55
            elif io:                   col,lw,al = C["edge_open"],blw*1.1,0.40
            else:                      col,lw,al = C["edge_base"],blw*0.8,0.18

            ax_g.plot([x0,x1],[y0,y1], color=col, linewidth=lw,
                      alpha=al, solid_capstyle="round", zorder=2)

            if ip and e["name"]:
                mx,my = (x0+x1)/2,(y0+y1)/2
                ax_g.text(mx, my, e["name"][:22], fontsize=4.5,
                          color=C["edge_path"], ha="center", va="center",
                          fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.1",
                                    fc=C["bg"], ec="none", alpha=0.85),
                          zorder=8)

        # ── Nodes (batch-draw for speed) ─────────────────────────
        xi,yi=[],[]; xo,yo=[],[]; xc,yc=[],[]
        for nid,n in net.nodes.items():
            x,y = self._xy(nid)
            if nid==src:
                ax_g.scatter(x,y,s=340,color=C["src"],edgecolors="#fff",linewidths=2,zorder=12)
                ax_g.scatter(x,y,s=800,color=C["src"],alpha=0.12,zorder=11)
                ax_g.text(x,y+0.045,n["name"],fontsize=7,color=C["src"],
                          ha="center",fontfamily="monospace",fontweight="bold",zorder=14)
            elif nid==dst:
                ax_g.scatter(x,y,s=340,color=C["dst"],edgecolors="#fff",linewidths=2,zorder=12)
                ax_g.scatter(x,y,s=800,color=C["dst"],alpha=0.12,zorder=11)
                ax_g.text(x,y+0.045,n["name"],fontsize=7,color=C["dst"],
                          ha="center",fontfamily="monospace",fontweight="bold",zorder=14)
            elif is_final and nid in path_nodes:
                ax_g.scatter(x,y,s=200,color=C["path"],
                             edgecolors=C["border"],linewidths=1.5,zorder=10)
                ax_g.text(x,y+0.038,n["name"],fontsize=5.5,color=C["path"],
                          ha="center",fontfamily="monospace",zorder=13)
            elif nid==current and not is_final:
                ax_g.scatter(x,y,s=280,color=C["current"],edgecolors="#fff",linewidths=2,zorder=11)
                ax_g.scatter(x,y,s=700,color=C["current"],alpha=0.15,zorder=10)
            elif nid in open_set: xo.append(x); yo.append(y)
            elif nid in closed:   xc.append(x); yc.append(y)
            else:                 xi.append(x); yi.append(y)

        if xi: ax_g.scatter(xi,yi,s=18,color=C["node_idle"],zorder=3,alpha=0.45)
        if xc: ax_g.scatter(xc,yc,s=35,color=C["closed"],  zorder=6,alpha=0.80)
        if xo: ax_g.scatter(xo,yo,s=55,color=C["open"],    zorder=7,alpha=0.90)

        # ── Title ────────────────────────────────────────────────
        if is_final:
            ax_g.set_title(
                f"PATH FOUND  |  {final_cost:.1f}s ({final_cost/60:.2f} min)"
                f"  |  {path_len} nodes  |  {net.city}, {net.state}",
                color=C["edge_path"], fontsize=9,
                fontfamily="monospace", pad=8)
        else:
            ax_g.set_title(
                f"Step {step_num}/{total}  |  "
                f"Open:{len(open_set)}  Closed:{len(closed)}  |  "
                f"{net.city}, {net.state}",
                color=C["text"], fontsize=8.5,
                fontfamily="monospace", pad=8)

        ax_g.legend(handles=[
            mpatches.Patch(color=C["src"],     label="Source"),
            mpatches.Patch(color=C["dst"],     label="Destination"),
            mpatches.Patch(color=C["current"], label="Current"),
            mpatches.Patch(color=C["open"],    label="Open (frontier)"),
            mpatches.Patch(color=C["closed"],  label="Closed (visited)"),
            mpatches.Patch(color=C["path"],    label="Optimal path"),
        ], loc="lower left", fontsize=6.5,
           facecolor=C["panel"], edgecolor=C["border"],
           labelcolor=C["text"], framealpha=0.95)

        # ── Log ──────────────────────────────────────────────────
        ax_l.set_xlim(0,1); ax_l.set_ylim(0,1)
        ax_l.set_title("Algorithm Log", color=C["text"],
                       fontsize=8.5, fontfamily="monospace", pad=6)
        mx=30; show=log_lines[-mx:]; lh=1.0/(mx+1)
        for i,(ev,msg) in enumerate(show):
            yp = 1.0-(i+1)*lh
            lbl = f"{self.EP.get(ev,'')} {msg}"
            if len(lbl)>46: lbl=lbl[:43]+"..."
            ax_l.text(0.03, yp, lbl, fontsize=6.5,
                      color=self.EC.get(ev,C["text"]),
                      fontfamily="monospace", va="center",
                      transform=ax_l.transAxes)

    # ── Live animation ────────────────────────────────────────────
    def _on_run(self, _):
        if self.net is None:
            with self.w_out: clear_output(); print("Load a city first (Step 1).")
            return
        src   = self.w_src.value
        dst   = self.w_dst.value
        delay = float(self.w_delay.value)
        if src == dst:
            with self.w_out: clear_output(); print("Source and destination must differ.")
            return

        engine = AStarEngine(self.net, src, dst)
        steps  = engine.steps
        total  = len(steps)

        pes = set()
        if engine.path:
            for i in range(len(engine.path)-1):
                u,v = engine.path[i],engine.path[i+1]
                pes.add((min(u,v), max(u,v)))

        log_lines = []
        with self.w_out:
            clear_output(wait=True)
            fig,(ax_g,ax_l) = plt.subplots(1,2,figsize=(16,7),
                gridspec_kw={"width_ratios":[2.5,1]})
            fig.patch.set_facecolor(self.C["bg"])
            plt.tight_layout(pad=1.8)

            for idx,step in enumerate(steps):
                ev   = step["event"]
                done = ev in ("found","no_path")
                log_lines.append((ev, step["msg"]))

                self._draw_frame(
                    ax_g, ax_l,
                    step["open"], step["closed"], step["current"],
                    src, dst, log_lines, idx+1, total,
                    set(engine.path) if done else set(),
                    pes if done else set(),
                    done, engine.cost or 0, len(engine.path))

                clear_output(wait=True)
                display(fig)
                plt.pause(0.001)
                time.sleep(delay*3 if done else delay)
                if done: break

            plt.close(fig)

        with self.w_out:
            print()
            if engine.path:
                n = self.net.nodes
                print(f"{'='*58}")
                print(f"  PATH FOUND  —  {self.net.city}, {self.net.state}")
                print(f"  Cost       : {engine.cost:.2f}s  ({engine.cost/60:.2f} min)")
                print(f"  Path nodes : {len(engine.path)}")
                print(f"  A* steps   : {total}")
                print(f"\n  Route:")
                for i,nid in enumerate(engine.path):
                    tag = "START" if i==0 else ("END  " if i==len(engine.path)-1 else f"  {i}  ")
                    nd = n[nid]
                    print(f"    [{tag}]  {nd['name']:28s}  ({nd['lat']:.4f},{nd['lon']:.4f})")
                print(f"{'='*58}")
            else:
                print("  NO PATH FOUND. Try different source/destination nodes.")

    # ── Static path ───────────────────────────────────────────────
    def _on_static(self, _):
        if self.net is None:
            with self.w_out: clear_output(); print("Load a city first.")
            return
        src,dst = self.w_src.value, self.w_dst.value
        engine  = AStarEngine(self.net, src, dst)
        pes = set()
        if engine.path:
            for i in range(len(engine.path)-1):
                u,v=engine.path[i],engine.path[i+1]; pes.add((min(u,v),max(u,v)))
        final     = engine.steps[-1]
        log_lines = [(s["event"],s["msg"]) for s in engine.steps]
        with self.w_out:
            clear_output(wait=True)
            fig,(ax_g,ax_l)=plt.subplots(1,2,figsize=(16,7),
                gridspec_kw={"width_ratios":[2.5,1]})
            fig.patch.set_facecolor(self.C["bg"])
            self._draw_frame(ax_g,ax_l,
                final["open"],final["closed"],None,src,dst,
                log_lines,len(engine.steps),len(engine.steps),
                set(engine.path),pes,True,engine.cost or 0,len(engine.path))
            plt.tight_layout(pad=1.8); plt.show()

    # ── Edge table ────────────────────────────────────────────────
    def _on_table(self, _):
        if self.net is None:
            with self.w_out: clear_output(); print("Load a city first.")
            return
        src,dst = self.w_src.value, self.w_dst.value
        engine  = AStarEngine(self.net, src, dst)
        with self.w_out:
            clear_output(wait=True)
            if not engine.path: print("No path found."); return
            rows = []
            for i in range(len(engine.path)-1):
                u,v = engine.path[i],engine.path[i+1]
                e   = next((ed for ed in self.net.edges if ed["u"]==u and ed["v"]==v),None)
                if e:
                    rows.append({
                        "Step":i+1,
                        "From":self.net.nodes[u]["name"][:22],
                        "To":  self.net.nodes[v]["name"][:22],
                        "Road":e["name"][:24],
                        "Type":e["highway"],
                        "Len(m)":int(e["length"]),
                        "Spd":e["speed_kph"],
                        "Base_t":round(e["travel_time"],1),
                        "Vol":int(e["traffic_volume"]),
                        "VC":e["vc_ratio"],
                        "Cong_t":round(e["congested_time"],2),
                        "Class":e["congestion_class"],
                    })
            df = pd.DataFrame(rows)
            sn = self.net.nodes[src]; dn = self.net.nodes[dst]
            display(HTML(
                f'<div style="font-family:monospace;background:#0d1117;'
                f'color:#58a6ff;padding:10px 16px;border-radius:6px;'
                f'border:1px solid #30363d;margin-bottom:10px;">'
                f'<b>{self.net.city}, {self.net.state}</b><br>'
                f'{sn["name"]} &rarr; {dn["name"]}<br>'
                f'<span style="color:#3fb950">Cost:{engine.cost:.2f}s '
                f'({engine.cost/60:.2f}min) &nbsp;&middot;&nbsp; '
                f'{len(engine.path)} nodes</span></div>'))
            display(df.style
                .set_properties(**{"background-color":"#161b22","color":"#c9d1d9",
                    "border":"1px solid #30363d","font-family":"monospace","font-size":"12px"})
                .set_table_styles([{"selector":"th","props":[
                    ("background-color","#0d1117"),("color","#58a6ff"),
                    ("font-family","monospace"),("font-size","12px"),
                    ("border","1px solid #30363d")]}])
                .bar(subset=["VC"],   color="#d29922",vmin=0,vmax=1.5)
                .bar(subset=["Cong_t"],color="#1f6feb")
                .format({"VC":"{:.3f}","Cong_t":"{:.2f}","Base_t":"{:.1f}"})
            )
            base=df["Base_t"].sum(); cong=df["Cong_t"].sum()
            worst=df.loc[df["VC"].idxmax()]
            print(f"\n{'─'*58}")
            print(f"  Base travel time    : {base:.2f} s")
            print(f"  Congested time      : {cong:.2f} s")
            print(f"  Traffic delay       : {cong-base:.2f} s")
            print(f"  Avg VC ratio        : {df['VC'].mean():.3f}")
            print(f"  Most congested road : {worst['Road']}  (VC={worst['VC']:.3f})")
            print(f"{'─'*58}")
