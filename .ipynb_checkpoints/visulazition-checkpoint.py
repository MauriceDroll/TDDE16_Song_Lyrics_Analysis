import pandas as pd
import plotly.graph_objects as go

# CSV-Datei laden
csv_file = 'topics.csv'  # Ersetzen Sie dies durch den Pfad zu Ihrer Datei
data = pd.read_csv(csv_file)

# Knoten erstellen
all_nodes = list(set(data['source']).union(set(data['target'])))
node_indices = {node: idx for idx, node in enumerate(all_nodes)}

# Quellen, Ziele und Werte zu Indizes konvertieren
data['source_idx'] = data['source'].map(node_indices)
data['target_idx'] = data['target'].map(node_indices)

# Sankey-Diagramm erstellen
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes
    ),
    link=dict(
        source=data['source_idx'],
        target=data['target_idx'],
        value=data['value']
    )
)])

# Titel und Layout anpassen
fig.update_layout(title_text="Sankey-Diagramm: LDA vs. BERT", font_size=12)
fig.show()
