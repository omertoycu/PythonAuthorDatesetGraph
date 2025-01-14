import webbrowser
import pandas as pd
from pyvis.network import Network
import ast  # String'i listeye dönüştürmek için kullanılacak
import os
from flask import Flask, request, render_template_string, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Global değişkenler
queue = []
nodes = {}
edges = {}

# Excel'den veri okuma fonksiyonu
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    required_columns = ['orcid', 'doi', 'author_position', 'author_name', 'coauthors', 'paper_title']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Excel dosyasında '{col}' sütunu eksik!")
    return df[required_columns]

data = read_excel_data('dataset.xlsx')

# Tüm yazarların toplam makale sayısını hesaplayan fonksiyon
def calculate_author_total_papers(data):
    author_total_papers = {}
    for _, row in data.iterrows():
        try:
            coauthors = ast.literal_eval(row['coauthors'])  # Coauthors listesini parse et
        except (ValueError, SyntaxError):
            coauthors = []
        coauthors = [coauthor.strip() for coauthor in coauthors if coauthor]  # Boş alanları temizle
        main_author = row['author_name'].strip()  # Ana yazarın ismi
        all_authors = set(coauthors)
        all_authors.add(main_author)
        for author in all_authors:
            author_total_papers[author] = author_total_papers.get(author, 0) + 1
    return author_total_papers


def calculate_node_weights(data, author_input):
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Tüm yazarların toplam makale sayısını hesapla
    author_total_papers = calculate_author_total_papers(data)
    logging.debug(f"Author total papers: {author_total_papers}")

    # A yazarı ile işbirliği yapan diğer yazarları belirle
    coauthors = set()
    for _, row in data.iterrows():
        try:
            if author_input in row['author_name'].strip().lower() or author_input in row['orcid'].strip().lower():
                row_coauthors = ast.literal_eval(row['coauthors'])
                coauthors.update([coauthor.strip() for coauthor in row_coauthors])
        except (ValueError, SyntaxError) as e:
            logging.error(f"Error parsing coauthors for {author_input}: {e}")
            continue

    # A yazarı ile işbirliği yapan yazarların makale sayısını al
    node_weights = []
    for coauthor in coauthors:
        if coauthor in author_total_papers:
            node_weights.append((coauthor, author_total_papers[coauthor]))

    # Düğüm ağırlıklarına göre sıralama
    node_weights.sort(key=lambda x: x[1], reverse=True)  # Ağırlığa göre azalan sırada sıralanır

    logging.debug(f"Node weights: {node_weights}")
    return node_weights


@app.route('/calculate_node_weights/<author_input>', methods=['GET'])
def calculate_node_weights_route(author_input):
    try:
        weights = calculate_node_weights(data, author_input)
        session['queue'] = weights  # Kuyruğu session'da sakla
        return jsonify({'weights': [{'name': author, 'articles': count} for author, count in weights]})
    except Exception as e:
        import logging
        logging.error(f"Error in calculate_node_weights_route for {author_input}: {e}")
        return jsonify({'error': str(e)})


@app.route('/dequeue_node', methods=['POST'])
def dequeue_node():
    queue = session.get('queue', [])
    author_input = request.form.get('author_id')

    if not queue:
        return jsonify({'message': 'Kuyruk boş!'})

    # Kuyruktan elemanı çıkar
    updated_queue = [item for item in queue if item[0].lower() != author_input.lower()]

    if len(updated_queue) == len(queue):  # Eğer hiçbir eleman çıkarılmadıysa
        return jsonify({'message': f"{author_input} kuyrukta bulunamadı."})

    session['queue'] = updated_queue
    remaining_authors = [{'name': author, 'articles': count} for author, count in updated_queue]

    return jsonify({'message': f"{author_input} kuyruktan çıkarıldı.", 'remaining_queue': remaining_authors})


# En çok işbirliği yapan yazarı belirleyen fonksiyon
def find_most_collaborative_author(edges):
    collaboration_count = {}
    for (source, target), weight in edges.items():
        collaboration_count[source] = collaboration_count.get(source, 0) + weight
        collaboration_count[target] = collaboration_count.get(target, 0) + weight
    most_collaborative_author = max(collaboration_count, key=collaboration_count.get)
    max_collaborations = collaboration_count[most_collaborative_author]
    return most_collaborative_author, max_collaborations


def create_manual_graph(data):
    global nodes, edges
    nodes = {}
    edges = {}
    main_authors = set()

    orcid_dict = {}  # ORCID'leri takip eden dict

    for _, row in data.iterrows():
        author = row['author_name'].strip()
        orcid = row['orcid'].strip()
        paper_title = row['paper_title']
        doi = row['doi']

        try:
            coauthors = ast.literal_eval(row['coauthors'])
        except (ValueError, SyntaxError):
            coauthors = []
        coauthors = [coauthor.strip() for coauthor in coauthors]

        # ORCID'e göre kontrol ve tek düğüm oluşturma
        if orcid not in orcid_dict:
            node_id = f"{author}-{orcid}"
            nodes[node_id] = {
                "label": author,
                "orcid": orcid,
                "papers": {},
                "color": "orange"
            }
            orcid_dict[orcid] = node_id
            main_authors.add(node_id)
        else:
            node_id = orcid_dict[orcid]

        if "papers" not in nodes[node_id]:
            nodes[node_id]["papers"] = {}
        nodes[node_id]["papers"][doi] = paper_title

        for coauthor in coauthors:
            if not coauthor or coauthor == author:  # Ana yazarın ismi yardımcı yazarlar arasında yer almamalı
                continue
            coauthor_id = coauthor
            if coauthor_id not in nodes:
                nodes[coauthor_id] = {
                    "label": coauthor,
                    "orcid": "N/A",
                    "papers": {},
                    "color": "lightblue"
                }
            nodes[coauthor_id]["papers"][doi] = paper_title  # Makale bilgisi ekle
            edge = tuple(sorted((node_id, coauthor_id)))
            if edge not in edges:
                edges[edge] = 0
            edges[edge] += 1

    # Toplam düğüm ve kenar sayısını yazdır (basit bir yazdırma ifadesiyle değiştirdik)
    print("Ana yazar sayısı:", len(main_authors))
    print("Toplam düğüm sayısı:", len(nodes))
    print("Toplam kenar sayısı:", len(edges))

    return nodes, edges, main_authors

def visualize_graph_with_output(nodes, edges, author_total_papers, highlight_path=None):
    graph = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    graph.set_options(""" 
    var options = {
      "nodes": {
        "scaling": {
          "min": 10,
          "max": 50
        }
      },
      "edges": {
        "arrows": {
          "to": false
        },
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -300,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.03
        },
        "minVelocity": 0.5,
        "timestep": 0.35
      }
    }
    """)

    avg_paper_count = sum(author_total_papers.values()) / len(author_total_papers)
    for node_id, node_data in nodes.items():
        paper_count = len(node_data["papers"])  # Düğümdeki makale sayısını kullan
        paper_info = node_data.get("papers", {})
        paper_details = "<br>".join(
            [f"<b>DOI:</b> {doi} - <b>Başlık:</b> {title}" for doi, title in paper_info.items()]
        )
        if paper_count > avg_paper_count * 1.2:
            size = 80
            color = "darkorange"
        elif paper_count < avg_paper_count * 0.8:
            size = 40
            color = "lightblue"
        else:
            size = 60
            color = "gray"

        if highlight_path and node_id in highlight_path:
            color = "red"
            size = 100

        graph.add_node(
            node_id,
            label=node_data['label'],
            title=f"""
                <b>Yazar:</b> {node_data['label']}<br>
                <b>ORCID:</b> {node_data['orcid']}<br>
                <b>Toplam Makale Sayısı:</b> {paper_count}<br>
                <b>Makaleler:</b><br>{paper_details if paper_details else 'Bu yazar için makale bilgisi yok.'}
            """,
            color=color,
            size=size
        )
    for (source, target), weight in edges.items():
        graph.add_edge(source, target, value=weight, width=weight * 2, title=f"Ortak Makale Sayısı: {weight}",
                       color="white")

    # En kısa yol veya en uzun yolu vurgulama ve zoom ekleme
    if highlight_path:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        for edge in path_edges:
            graph.add_edge(edge[0], edge[1], color='red', width=2)

    return graph


# En kısa yolu BFS uygulanarak bulan fonksiyon
def find_shortest_path_between_authors(start_author, end_author, edges):
    # BFS için bir kuyruk ve ziyaret edilen düğümler listesi
    queue = [(start_author, [start_author])]
    visited = set()

    while queue:
        current_node, path = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        # Eğer hedef düğüme ulaşırsak yolu döndür
        if current_node == end_author:
            return path

        # Komşu düğümleri sıraya ekle
        for edge in edges:
            if current_node in edge:
                neighbor = edge[0] if edge[1] == current_node else edge[1]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    # Eğer hedef düğüme ulaşılamazsa boş liste döndür
    return []


def calculate_shortest_paths_from_author(author_id, nodes, edges):
    distances = {node_id: float('inf') for node_id in nodes}
    previous_nodes = {node_id: None for node_id in nodes}
    distances[author_id] = 0
    unvisited_nodes = set(nodes.keys())

    while unvisited_nodes:
        # En kısa mesafeye sahip düğümü bul
        current_node = min(unvisited_nodes, key=lambda node: distances[node])
        unvisited_nodes.remove(current_node)

        if distances[current_node] == float('inf'):
            break

        # Komşu düğümler üzerinden güncellemeler yap
        for neighbor in get_neighbors(current_node, edges):
            if neighbor in unvisited_nodes:
                # Kenar yönünü kontrol et
                edge_key = (current_node, neighbor) if (current_node, neighbor) in edges else (neighbor, current_node)
                distance = distances[current_node] + edges[edge_key]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node

    return distances, previous_nodes

def get_neighbors(node, edges):
    neighbors = []
    for (source, target) in edges:
        if source == node:
            neighbors.append(target)
        elif target == node:
            neighbors.append(source)
    return neighbors

@app.route('/shortest_paths_from_author', methods=['POST'])
def shortest_paths_from_author():
    author_input = request.form.get('author_id')
    author_id = find_node_id(author_input)

    if not author_id:
        return jsonify({'error': f"Yazar '{author_input}' bulunamadı."})

    distances, previous_nodes = calculate_shortest_paths_from_author(author_id, nodes, edges)

    # Adım adım tabloyu güncelleme
    result_text = f"{nodes[author_id]['label']} için en kısa yollar:\n"
    for node_id, distance in distances.items():
        if distance < float('inf'):
            path = []
            current = node_id
            while current is not None:
                path.append(nodes[current]['label'])
                current = previous_nodes[current]
            path = ' ➔ '.join(path[::-1])
            result_text += f"Yazar: {nodes[node_id]['label']}, Mesafe: {distance}, Yol: {path}\n"

    return result_text


def find_node_id(author_input):
    author_input = author_input.strip().lower()
    return next((node_id for node_id, node_data in nodes.items()
                 if author_input in node_id.lower() or author_input in node_data['label'].lower() or author_input == node_data['orcid'].lower()), None)

@app.route('/shortest_path/<start_author>/<end_author>', methods=['GET'])
def shortest_path_route(start_author, end_author):
    start_author = start_author.strip().lower()
    end_author = end_author.strip().lower()

    # Kullanıcı girdisine göre düğüm ID'lerini bul
    start_node = find_node_id(start_author)
    end_node = find_node_id(end_author)

    if not start_node or not end_node:
        return f"Hata: '{start_author}' veya '{end_author}' girdisiyle eşleşen bir yazar bulunamadı."

    # En kısa yol algoritmasını çağır
    queue = find_shortest_path_between_authors(start_node, end_node, edges)

    if not queue:
        return f"{start_author} ile {end_author} arasında bir bağlantı bulunamadı."

    session['queue'] = queue

    # Yol detaylarını döndür
    path_details = " ➔ ".join([nodes[node]['label'] for node in queue])
    path_length = len(queue)

    # Grafiği oluştur ve HTML dosyasını kaydet
    graph = visualize_graph_with_output(nodes, edges, calculate_author_total_papers(data), queue)
    graph_path = f"graph_{start_author}_{end_author}.html"
    graph.save_graph(graph_path)

    # Yeni pencerede grafiği aç
    webbrowser.open_new_tab(graph_path)

    return f"En kısa yol: {path_details} (Uzunluk: {path_length})"


# Kullanıcıdan gelen yazar ID'sine göre işbirliği sayısını hesaplama route'u
@app.route('/calculate_collaborators/<author_input>', methods=['GET'])
def calculate_collaborators(author_input):
    author_input = author_input.strip().lower()

    node_id = find_node_id(author_input)

    if not node_id:
        return f"Hata: '{author_input}' girdisiyle eşleşen bir yazar bulunamadı."

    collaborators = set()

    for (source, target) in edges:
        if source == node_id or target == node_id:
            collaborators.add(source)
            collaborators.add(target)

    collaborators.discard(node_id)

    total_collaborators = len(collaborators)

    return f"'{author_input}' yazarının işbirliği yaptığı toplam yazar sayısı: {total_collaborators}"

def find_longest_path(start_node, visited=None, current_path=None):
    if visited is None:
        visited = set()
    if current_path is None:
        current_path = []

    visited.add(start_node)
    current_path.append(start_node)

    longest_path = current_path.copy()
    for (source, target) in edges:
        if source == start_node and target not in visited:
            new_path = find_longest_path(target, visited, current_path)
            if len(new_path) > len(longest_path):
                longest_path = new_path
        elif target == start_node and source not in visited:
            new_path = find_longest_path(source, visited, current_path)
            if len(new_path) > len(longest_path):
                longest_path = new_path

    visited.remove(start_node)
    current_path.pop()

    return longest_path



# BST düğüm sınıfı ve ekleme/silme fonksiyonları
class TreeNode:
    def __init__(self, author_name):
        self.author_name = author_name
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, author_name):
        if not self.root:
            self.root = TreeNode(author_name)
        else:
            self._insert(self.root, author_name)

    def _insert(self, node, author_name):
        if author_name < node.author_name:
            if node.left is None:
                node.left = TreeNode(author_name)
            else:
                self._insert(node.left, author_name)
        elif author_name > node.author_name:
            if node.right is None:
                node.right = TreeNode(author_name)
            else:
                self._insert(node.right, author_name)

    def delete(self, author_name):
        self.root = self._delete(self.root, author_name)

    def _delete(self, node, author_name):
        if not node:
            return node

        if author_name < node.author_name:
            node.left = self._delete(node.left, author_name)
        elif author_name > node.author_name:
            node.right = self._delete(node.right, author_name)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            temp_val = self._min_value_node(node.right)
            node.author_name = temp_val.author_name
            node.right = self._delete(node.right, temp_val.author_name)

        return node

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder_traversal(self, node, result):
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.author_name)
            self.inorder_traversal(node.right, result)
        return result

# Kuyruktaki yazarları döndüren fonksiyon
def get_authors_queue(data):
    authors_queue = []
    for _, row in data.iterrows():
        main_author = row['author_name'].strip().lower()
        authors_queue.append(main_author)
    return list(set(authors_queue))  # Tekrar eden yazarları çıkartalım

# BST'yi kuyruktan oluşturan fonksiyon
def create_bst_from_queue(queue):
    bst = BinarySearchTree()
    for author in queue:
        bst.insert(author)
    return bst

# BST'yi görselleştiren fonksiyon
def visualize_bst(bst):
    graph = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

    def add_nodes(node):
        if node:
            color = "red"
            graph.add_node(node.author_name, label=node.author_name, color=color)
            add_nodes(node.left)
            add_nodes(node.right)

    def add_edges(node):
        if node:
            if node.left:
                graph.add_edge(node.author_name, node.left.author_name, color="white")
                add_edges(node.left)
            if node.right:
                graph.add_edge(node.author_name, node.right.author_name, color="white")
                add_edges(node.right)

    add_nodes(bst.root)

    add_edges(bst.root)

    html_content = graph.generate_html()

    output_file = "bst.html"
    with open(output_file, 'w') as f:
        f.write(html_content)

    webbrowser.open_new_tab(output_file)

@app.route('/create_bst', methods=['POST'])
def create_bst_route():
    queue = session.get('queue', [])

    if not queue:
        return "Kuyruk oluşturulamadı!"

    bst = create_bst_from_queue(queue)
    author_id_to_delete = request.form.get('author_id')
    if author_id_to_delete:
        bst.delete(author_id_to_delete)

    author_details = bst.inorder_traversal(bst.root, [])

    visualize_bst(bst)

    output = "BST oluşturuldu. \nYazar İsimleri:\n"
    for author_name in author_details:
        output += f"{author_name}\n"

    return output


@app.route('/longest_path', methods=['POST'])
def get_longest_path():
    author_id = request.form.get('author_id')
    author_id = author_id.strip().lower()

    # Yazar ID'sini veya ORCID'i düğümler arasında bulalım
    start_node = None
    for node_id, node_data in nodes.items():
        if author_id in node_id.lower() or author_id == node_data['orcid'].lower():
            start_node = node_id
            break

    if not start_node:
        return f"Geçersiz veya bulunamayan yazar ID: {author_id}"

    # Toplam makale sayısını hesapla
    author_total_papers = calculate_author_total_papers(data)

    longest_path = find_longest_path(start_node)
    path_labels = " ➔ ".join([nodes[node]['label'] for node in longest_path])
    path_length = len(longest_path)

    # Grafiği oluştur ve HTML dosyasını kaydet
    graph = visualize_graph_with_output(nodes, edges, author_total_papers, highlight_path=longest_path)
    graph_path = f"graph_longest_path_{author_id}.html"
    graph.save_graph(graph_path)

    # Yeni pencerede grafiği aç
    webbrowser.open_new_tab(graph_path)

    return f"Yazar {nodes[start_node]['label']} için en uzun yol: {path_labels} (Uzunluk: {path_length})"


@app.route('/', methods=['GET', 'POST'])
def wish_me_luck():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            data = read_excel_data(file_path)
            author_total_papers = calculate_author_total_papers(data)
            nodes, edges, main_authors = create_manual_graph(data)
            graph = visualize_graph_with_output(nodes, edges, author_total_papers)
            most_collaborative_author, max_collaborations = find_most_collaborative_author(edges)
            # HTML şablonunu oluştur
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Interactive Graph Visualization</title>
                <style>
                    body {{
                        display: flex;
                        margin: 0;
                        font-family: Arial, sans-serif;
                    }}
                    #sidebar {{
                        width: 20%;
                        background: #333;
                        color: white;
                        display: flex;
                        flex-direction: column;
                        padding: 10px;
                        height: 100vh;
                        box-sizing: border-box;
                    }}
                    #sidebar h3 {{
                        text-align: center;
                        margin-bottom: 10px;
                    }}
                    #buttons {{
                        flex: 1;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-evenly;
                    }}
                    .button {{
                        margin: 5px 0;
                        padding: 10px;
                        text-align: center;
                        background: #444;
                        color: white;
                        border: 1px solid white;
                        cursor: pointer;
                        border-radius: 5px;
                    }}
                    .button:hover {{
                        background: #555;
                    }}
                    #output {{
                        flex: 1;
                        background: #222;
                        color: white;
                        padding: 10px;
                        overflow-y: auto;
                        border: 1px solid #444;
                        border-radius: 5px;
                        height: 30%;
                    }}
                    #graph-container {{
                        width: 80%;
                        height: 100vh;
                    }}
                </style>
                <script>
                    function updateOutput(content) {{
                        document.getElementById('output').innerHTML = content;
                    }}

                    function showMostCollaborativeAuthor() {{
                        const content = "En çok işbirliği yapan yazar: {most_collaborative_author} (Toplam İşbirliği: {max_collaborations})";
                        updateOutput(content);
                    }}


                    function calculateNodeWeights() {{
            const authorInput = prompt("Lütfen A yazarının ID'sini veya ismini giriniz:");
            fetch(`/calculate_node_weights/${{authorInput}}`)
                .then(response => response.json())
                .then(data => {{
                    let content = 'Düğüm ağırlıkları hesaplandı.<br>';
                    data.weights.forEach(weight => {{
                        content += `${{weight.name}}: ${{weight.articles}} makale<br>`;
                    }});
                    updateOutput(content);
                }})
                .catch(error => updateOutput('Bir hata oluştu: ' + error));
        }}

        function dequeueNode() {{
            const authorInput = prompt("Lütfen kuyruktan çıkarmak isteğiniz yazarın ismini giriniz:");
            const formData = new FormData();
            formData.append('author_id', authorInput);
            fetch('/dequeue_node', {{
                method: 'POST',
                body: formData
            }})
            .then(response => response.json())
            .then(data => {{
                let content = data.message + '<br>Kalan kuyruk:<br>';
                data.remaining_queue.forEach(weight => {{
                    content += `${{weight.name}}: ${{weight.articles}} makale<br>`;
                }});
                updateOutput(content);
            }})
            .catch(error => updateOutput('Bir hata oluştu: ' + error));
        }}

                    function calculateShortestPathsFromAuthor() {{
            const authorInput = prompt("Lütfen en kısa yollarını hesaplamak istediğiniz yazar ID'sini veya ismini girin:");
            if (!authorInput) {{
                updateOutput("Herhangi bir ID girilmedi!");
                return;
            }}
            const formData = new FormData();
            formData.append('author_id', authorInput);
            fetch('/shortest_paths_from_author', {{
                method: 'POST',
                body: formData
            }})
            .then(response => response.text())
            .then(data => {{
                updateOutput(data);
            }})
            .catch(error => updateOutput("Bir hata oluştu: " + error.message));
        }}

                     function calculateShortestPath() {{
                        const startAuthor = prompt("Lütfen başlangıç yazarının ID'sini veya adını girin:");
                        const endAuthor = prompt("Lütfen bitiş yazarının ID'sini veya adını girin:");
                        if(!(startAuthor && endAuthor)) {{
                            updateOutput("Herhangi bir ID girilmedi!");
                            return;
                        }}
                        fetch(`/shortest_path/${{startAuthor}}/${{endAuthor}}`)
                            .then(response => response.text())
                            .then(data => updateOutput(data))
                            .catch(error => updateOutput('Bir hata oluştu: ' + error));
                   }}

                    function calculateCollaborators() {{
                        const authorId = prompt("Lütfen işbirliği sayısını hesaplamak istediğiniz yazarın ID'sini girin:");
                        if (!authorId) {{
                            updateOutput("Herhangi bir ID girilmedi!");
                            return;
                        }}
                        fetch(`/calculate_collaborators/${{authorId}}`)
                            .then(response => response.text())
                            .then(data => updateOutput(data))
                            .catch(error => updateOutput('Bir hata oluştu: ' + error));
                    }}

                    function createBSTFromQueue() {{
                        const authorId = prompt("Lütfen silmek isteğiniz yazarın ID'sini girin:");
                        if (!authorId) {{
                            updateOutput("Herhangi bir ID girilmedi!");
                            return;
                        }}
                        const formData = new FormData();
                        formData.append('author_id', authorId);
                        fetch('/create_bst', {{
                            method: 'POST',
                            body: formData
                        }})
                        .then(response => response.text())
                        .then(data => updateOutput(data))
                        .catch(error => updateOutput('Bir hata oluştu: ' + error));
                    }}

                    function LongestPath() {{
                        const authorId = prompt("En uzun yolunu bulmak isteğiniz yazarın ID'si giriniz:");
                        if (!authorId) {{
                            updateOutput("Herhangi bir ID girilmedi!");
                            return;
                        }}
                        fetch('/longest_path', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/x-www-form-urlencoded',
                            }},
                            body: `author_id=${{authorId}}`,
                        }})
                        .then(response => response.text())
                        .then(data => {{
                           updateOutput(data);
                        }})
                        .catch(error => {{
                            updateOutput("Bir hata oluştu: " + error.message);
                        }});
                    }}
                </script>
            </head>
            <body>
                <div id="sidebar">
                    <h3>İster Menüsü</h3>
                    <div id="buttons">
                        <div class="button" onclick="calculateShortestPath()">1. A ile B yazarı arasındaki en kısa yol</div>
                        <div class="button" onclick="calculateNodeWeights()">2. A yazarı için düğüm ağırlıkları</div>
                        <div class="button" onclick="createBSTFromQueue()">3. Kuyruktan BST oluştur</div>
                        <div class="button" onclick="calculateShortestPathsFromAuthor()">4. Yazar için kısa yolları hesapla</div>
                        <div class="button" onclick="calculateCollaborators()">5. İşbirliği yapılan yazar sayısı</div>
                        <div class="button" onclick="showMostCollaborativeAuthor()">6. En çok işbirliği yapan yazar</div>
                        <div class="button" onclick="LongestPath()">7. Yazar için en uzun yolun bulunması</div>
                        <div class="button" onclick="dequeueNode()">8. Kuyruktan Düğüm Çıkar</div> <!-- Yeni buton eklendi -->
                    </div>
                    <div id="output">Çıktı Ekranı.</div>
                </div>
                <div id="graph-container">
                    {graph.generate_html()}
                </div>
            </body>
            </html>
            """
            return render_template_string(html_template)

    return '''
    <!doctype html>
    <title>Excel Dosyasını Yükle</title>
    <h1>Excel Dosyasını Yükle</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Yükle>
    </form>
    '''


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)


