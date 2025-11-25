import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import random
import heapq
import time

# =============================================
# PARÂMETROS
# =============================================
NODES = ["A", "B", "C", "D", "E", "F", "G"]
PROB_EXTRA_EDGE = 0.25
MIN_WEIGHT = 1
MAX_WEIGHT = 10

# =============================================
# FUNÇÃO: Criar Grafo Randomizado
# =============================================
def criar_grafo_randomizado():
    G = nx.Graph()
    G.add_nodes_from(NODES)

    # Garante que o grafo seja conectado
    shuffled = NODES[:]
    random.shuffle(shuffled)
    for i in range(len(shuffled) - 1):
        G.add_edge(shuffled[i], shuffled[i + 1], weight=random.randint(MIN_WEIGHT, MAX_WEIGHT))

    # Arestas extras randomizadas
    for i in range(len(NODES)):
        for j in range(i + 1, len(NODES)):
            if random.random() < PROB_EXTRA_EDGE and not G.has_edge(NODES[i], NODES[j]):
                G.add_edge(NODES[i], NODES[j], weight=random.randint(MIN_WEIGHT, MAX_WEIGHT))

    return G

# =============================================
# DESENHAR GRAFO
# =============================================
def draw_graph_ax(ax, G, pos, node_colors=None, edge_highlight=None, path_edges=None, title="Grafo"):
    ax.clear()

    default_node_color = "lightblue"
    node_color_list = [node_colors.get(n, default_node_color) if node_colors else default_node_color for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, node_size=900)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=1.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), ax=ax)

    if edge_highlight:
        for (u, v, color, lw) in edge_highlight:
            if G.has_edge(u, v):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color=color, width=lw)

    if path_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges, edge_color="orange", width=4)

    ax.set_title(title)
    ax.set_axis_off()

# =============================================
# ANIMAÇÃO DO DIJKSTRA
# =============================================
def dijkstra_animated(G, start, goal, pos, fig, ax, pause_step=0.6):

    dist = {n: float('inf') for n in G.nodes()}
    prev = {n: None for n in G.nodes()}
    dist[start] = 0

    visited = set()
    heap = [(0, start)]
    in_frontier = {start}

    node_colors = {n: "lightblue" for n in G.nodes()}
    node_colors[start] = "yellow"

    draw_graph_ax(ax, G, pos, node_colors=node_colors, title=f"Início: {start}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(pause_step)

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue

        visited.add(u)
        in_frontier.discard(u)
        node_colors[u] = "green"

        draw_graph_ax(ax, G, pos, node_colors=node_colors, title=f"Visitando: {u}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(pause_step)

        if u == goal:
            break

        for v in G.neighbors(u):
            if v in visited:
                continue

            w = G[u][v]["weight"]
            node_colors[v] = "yellow"

            draw_graph_ax(ax, G, pos, node_colors=node_colors, edge_highlight=[(u, v, "red", 3)], title=f"Processando: {u} → {v}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(pause_step * 0.5)

            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
                in_frontier.add(v)

    if dist[goal] == float('inf'):
        return None, None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    path_edges = list(zip(path, path[1:]))

    for n in G.nodes():
        if n in path:
            node_colors[n] = "orange"
        elif n in visited:
            node_colors[n] = "green"

    draw_graph_ax(ax, G, pos, node_colors=node_colors, path_edges=path_edges, title=f"Caminho final: {start} → {goal} (dist={dist[goal]})")
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)

    return dist[goal], path

# =============================================
# INTERFACE TKINTER
# =============================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Projeto Dijkstra – Tkinter + Matplotlib")
        self.root.geometry("1100x700")

        # Frame Superior (controles)
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Início:").grid(row=0, column=0)
        tk.Label(control_frame, text="Fim:").grid(row=0, column=2)

        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()

        self.start_box = ttk.Combobox(control_frame, textvariable=self.start_var, values=NODES, width=5)
        self.end_box = ttk.Combobox(control_frame, textvariable=self.end_var, values=NODES, width=5)

        self.start_box.grid(row=0, column=1, padx=5)
        self.end_box.grid(row=0, column=3, padx=5)

        tk.Button(control_frame, text="Gerar Novo Grafo", command=self.new_graph).grid(row=0, column=4, padx=15)
        tk.Button(control_frame, text="Executar Dijkstra", command=self.run_dijkstra).grid(row=0, column=5, padx=15)

        # Área do gráfico
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        # 🔸 ADICIONADO: Label para mostrar o resultado
        self.result_label = tk.Label(root, text="Caminho: ---   |   Distância: ---", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # Criar primeiro grafo
        self.new_graph()

    # ===== Criar novo grafo =====
    def new_graph(self):
        self.G = criar_grafo_randomizado()
        self.pos = nx.spring_layout(self.G, seed=42, k=1.5, iterations=300)
        draw_graph_ax(self.ax, self.G, self.pos, title="Grafo Randomizado")
        self.canvas.draw()

        # 🔸 limpa o texto ao gerar novo grafo
        self.result_label.config(text="Caminho: ---   |   Distância: ---")

    # ===== Executar Dijkstra =====
    def run_dijkstra(self):
        start = self.start_var.get()
        end = self.end_var.get()

        if not start or not end:
            print("Selecione início e fim.")
            return

        dist, path = dijkstra_animated(self.G, start, end, self.pos, self.fig, self.ax)
        self.canvas.draw()

        if dist is None:
            self.result_label.config(text="Nenhum caminho encontrado.")
            return

        caminho = " → ".join(path)
        self.result_label.config(text=f"Caminho: {caminho}   |   Distância: {dist}")

        

# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()