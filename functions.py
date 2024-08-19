import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Read excel file
def read_excel(file, sheet_name=0):
    """
    Reads an Excel file and returns a DataFrame from the specified sheet.

    Parameters:
    - file (str): Path to the Excel file.
    - sheet_name (str or int): Sheet name or index to read from. Default is the first sheet.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the specified sheet.
    """
    df = pd.read_excel(file, sheet_name=sheet_name)
    df = pd.DataFrame(df)
    return df

# Read csv file
def read_csv(file):
    df = pd.read_csv(file)
    df = pd.DataFrame(df)
    return df


# Clean data
columns = ['ledger_fee_id', 'employment', 'phone_number', 'success_redirect_url', 'given_name', 
            'business_nature', 'account_details', 'gender', 'expires_at', 'client', 'fee', 
            'hashed_phone_number', 'refunded_amount', 'payment_channel_transaction_id', 
            'linked_account_id', 'is_otp_required', 'otp_mobile_number', 'settlement_date', 
            'business_type', 'failure_code', 'callback_url', 'status', 'channel_account_reference', 
            'updated', 'device_fingerprint', 'date_of_birth', 'description', 'domicile_of_registration', 
            'version', 'checkout_url', 'vat', 'trading_name', 'client_type', 'id', 'status2', 
            'payment_channel_verification_id', 'meta', 'transacting_entity', 'internal_metadata', 
            'idempotency_key', 'nationality', 'ledger_transaction_id', 'business_name', 
            'connector_metadata', 'installment', 'time', 'given_names', 'business_id', 'basket', 
            'domicile_country', 'end_customer_id', 'created', 'middle_name', 'amount', 'email', 
            'channel_code', 'client_reference', 'ledger_payment_id', 'failure_redirect_url', 
            'given_names_non_roman', 'customer_id', 'required_action', 'surname', 
            'surname_non_roman', 'payment_channel_reference_id', 'entity', 'type', 'account_hash', 
            'date_of_registration', 'business_domicile', 'occupation', 'date_of_account_registration', 
            'ledger_settlement_id', 'dt', 'mother_maiden_name', 'account_type', 'bank_acc', 
            'otp_expiration_timestamp', 'place_of_birth', 'metadata', 'business_subtype', 
            'currency', 'payment_method_id', 'mobile_number', 'reference_id', 'enable_otp']

def preprocess_dataframe(df, colums_to_keep=columns):
    """
    Preprocesses a DataFrame by:
    - Making column names lowercase
    - Converting all text to lowercase
    - Replacing spaces with underscores in both column names and text values
    - Removing columns with all values missing
    - Keeping only specified columns

    Parameters:
    - df (pd.DataFrame): The DataFrame to preprocess.
    - colums_to_keep (list): List of columns to keep in the DataFrame.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Convert entire table to lowercase
    df = df.apply(lambda x: x.astype(str).str.lower())
    
    # Replace spaces with underscores in column names and text values
    df.columns = df.columns.str.replace(' ', '_')
    df = df.apply(lambda x: x.str.replace(' ', '_'))
    
    # Keep only specified columns
    df = df[colums_to_keep]   
    
    # Replace 'nan' strings with actual NaN values
    df.replace('nan', np.nan, inplace=True)
    
    # Remove columns with all values missing
    df = df.dropna(axis=1, how='all')
    


    return df

def describe_dataframe(df):
    """
    Provides a description of the DataFrame including:
    - Number of rows and columns
    - Column names and data types
    - Basic statistics (for numeric columns)
    - A preview of the first few rows
    - Number of NaN values per column, sorted by highest to lowest
    - Number of unique values per column

    Parameters:
    - df (pd.DataFrame): The DataFrame to describe.

    Returns:
    - None
    """
    if df.empty:
        print("The DataFrame is empty.")
        return
    
    # Number of rows and columns
    num_rows, num_cols = df.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    
    # Column names and data types
    print("\nColumn names and data types:")
    print(df.dtypes)
    
    # Number of NaN values per column, sorted by highest to lowest
    nan_counts = df.isna().sum().sort_values(ascending=False)
    print("\nNumber of NaN values per column (sorted by highest to lowest):")
    print(nan_counts)

    # Number of unique values per column
    unique_counts = df.nunique().sort_values(ascending=False)
    print("\nNumber of unique values per column (sorted by highest to lowest):")
    print(unique_counts)

    # Basic statistics for numeric columns
    if not df.select_dtypes(include=[np.number]).empty:
        print("\nBasic statistics for numeric columns:")
        print(df.describe(include=[np.number]))
    else:
        print("\nNo numeric columns available for basic statistics.")



def naive_bayes_conditional_probabilities(df, feature, value):
    """
    Calculate the conditional probabilities of a feature value given that a transaction is fraudulent
    and a transaction being fraudulent given the feature value using Naive Bayes.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the transaction data.
    feature (str): The feature for which to calculate the conditional probabilities.
    value: The value of the feature to calculate the conditional probabilities for.
    
    Returns:
    tuple: A tuple containing P(feature=value | is_fraud=1), P(is_fraud=1 | feature=value), and P(is_fraud=0 | feature=value).
    """
    # Calculate P(is_fraud=1)
    p_fraud = df['is_fraud'].mean()
    
    # Calculate P(is_fraud=0)
    p_not_fraud = 1 - p_fraud
    
    # Calculate P(feature=value | is_fraud=1)
    p_value_given_fraud = df[df['is_fraud'] == 1][feature].value_counts(normalize=True).get(value, 0)
    
    # Calculate P(feature=value | is_fraud=0)
    p_value_given_not_fraud = df[df['is_fraud'] == 0][feature].value_counts(normalize=True).get(value, 0)
    
    # Calculate P(feature=value)
    p_value = df[feature].value_counts(normalize=True).get(value, 0)
    
    # Calculate P(is_fraud=1 | feature=value) and P(is_fraud=0 | feature=value) using Bayes' theorem
    if p_value == 0:
        p_fraud_given_value = 0
        p_not_fraud_given_value = 0
    else:
        numerator_fraud = p_value_given_fraud * p_fraud
        numerator_not_fraud = p_value_given_not_fraud * p_not_fraud
        denominator = numerator_fraud + numerator_not_fraud
        
        p_fraud_given_value = numerator_fraud / denominator
        p_not_fraud_given_value = numerator_not_fraud / denominator
    
    return p_fraud_given_value, p_not_fraud_given_value


def factorize_and_map(df, token_mapping):
    """
    Factorizes columns in the DataFrame and applies token mapping.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to preprocess.
    - token_mapping (dict): Dictionary with column names as keys and token prefixes as values.
    
    Returns:
    - pd.DataFrame: The DataFrame with factorized and token-mapped columns.
    - dict: Dictionary containing the factorized values to original values mappings for each column.
    - dict: Dictionary containing token-mapped values to original values mappings for each column.
    """
    # Dictionary to store factorized to original mappings
    factorized_to_original = {}
    token_to_original = {}
    
    # Factorize and apply token mapping
    for col, prefix in token_mapping.items():
        if col in df.columns:
            # Factorize the column and store the mapping
            factorized_values, original_values = pd.factorize(df[col])
            factorized_to_original[col] = dict(enumerate(original_values))
            
            # Create token to original mapping
            token_mapped_values = {f"{prefix}{i}": original_values[i] for i in range(len(original_values))}
            token_to_original[col] = token_mapped_values
            
            # Apply factorization
            df[col] = factorized_values
            
            # Apply token mapping
            df[col] = df[col].apply(lambda x: f"{prefix}{x}")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    
    return df, factorized_to_original, token_to_original

def get_original_value_from_token(token_mapped_value, token_to_original):
    """
    Retrieves the original value from a token-mapped value using the global token_to_original mapping.
    
    Parameters:
    - token_mapped_value (str): The token-mapped value to recover.
    - token_to_original (dict): Dictionary containing token-mapped values to original values mappings for each column.
    
    Returns:
    - str: The original value corresponding to the token-mapped value.
    """
    for column_name, mapping in token_to_original.items():
        if token_mapped_value in mapping:
            return mapping[token_mapped_value]
    raise ValueError(f"Token-mapped value '{token_mapped_value}' not found in any column.")

def create_and_plot_network(df, node_cols, edge_cols, color_col, node_attr_cols):
    """
    Creates and plots a network graph from a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - node_cols (list): List of node columns (up to 2) to be used in the graph.
    - edge_cols (list): List of edge columns (should be exactly one column for weights).
    - color_col (str): Column name used for node colors.
    - node_attr_cols (list): List of columns to be used as node attributes.
    
    Returns:
    - G (nx.Graph): The created network graph.
    """
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_col in node_cols:
        for value in df[node_col].unique():
            # Combine all relevant attributes for nodes from the DataFrame
            attributes = {}
            for col in node_attr_cols:
                attributes[col] = df[df[node_col] == value][col].iloc[0]
            attributes['node_type'] = node_col
            G.add_node(value, **attributes)
    
    # Add edges with summed weights
    if len(edge_cols) != 1:
        raise ValueError("Exactly one edge column is required for weights.")
    
    weight_col = edge_cols[0]
    edge_weights = {}
    for _, row in df.iterrows():
        edge = (row[node_cols[0]], row[node_cols[1]])
        weight = row[weight_col]
        if edge in edge_weights:
            edge_weights[edge] += weight
        else:
            edge_weights[edge] = weight
    
    for edge, weight in edge_weights.items():
        G.add_edge(*edge, weight=weight)
    
    # Set node colors based on color_col
    if color_col:
        unique_values = df[color_col].unique()
        color_map = plt.get_cmap('hsv')
        value_to_color = {value: color_map(i / len(unique_values)) for i, value in enumerate(unique_values)}
        node_colors = [value_to_color[G.nodes[node].get(color_col, 'grey')] for node in G.nodes()]
    else:
        node_colors = ['grey'] * len(G.nodes())
    
    # Set node shapes based on node types
    unique_node_types = set(nx.get_node_attributes(G, 'node_type').values())
    shape_map = {node_type: shape for node_type, shape in zip(unique_node_types, ['o', '^'])}
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(24, 16))
    
    # Draw the graph
    pos = nx.spring_layout(G)
    
    # Draw nodes with different shapes and colors
    for node_type, shape in shape_map.items():
        shape_nodes = [node for node in G.nodes() if G.nodes[node].get('node_type') == node_type]
        nx.draw_networkx_nodes(G, pos, 
                                nodelist=shape_nodes,
                                node_color=[value_to_color[G.nodes[node].get(color_col, 'grey')] for node in shape_nodes],
                                node_shape=shape,
                                label=node_type,
                                ax=ax)
    
    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=2, style='dashed')
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, ax=ax)
    
    ax.set_title("Network Graph")
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return G

def filter_graph_by_attributes(G, attr_filters):
    """
    Filters the nodes in the graph G based on specified node attributes.
    
    Parameters:
    - G (nx.Graph): The original network graph.
    - attr_filters (dict): Dictionary of attribute filters. Keys are attribute names, and values are the values to filter by.
    
    Returns:
    - filtered_G (nx.Graph): The filtered network graph.
    """
    filtered_G = nx.Graph()
    
    # Add nodes that match the filter criteria
    for node, attributes in G.nodes(data=True):
        match = True
        for attr, value in attr_filters.items():
            if attributes.get(attr) != value:
                match = False
                break
        if match:
            filtered_G.add_node(node, **attributes)
    
    # Add edges between the filtered nodes
    for edge in G.edges(data=True):
        if edge[0] in filtered_G and edge[1] in filtered_G:
            filtered_G.add_edge(edge[0], edge[1], **edge[2])
    
    return filtered_G

def plot_subgraphs(G, nodes_per_page=6, node_size=70, num_hops=2, 
                    node_font_size=6, title_font_size=10, 
                    edge_label_font_size=6, figsize_per_row=5,
                    max_figures=10, color_attribute='email',
                    bins=20):
    """
    Plots subgraphs of a network graph, with settings for pagination, node size, 
    font sizes, and figure size. Limits the number of figures generated and calculates
    the distribution of the total sum of edge weights per subgraph.

    Parameters:
    - G (nx.Graph): The graph to plot.
    - nodes_per_page (int): Number of nodes per page.
    - node_size (int): Size of nodes.
    - num_hops (int): Number of hops to include in the neighborhood.
    - node_font_size (int): Font size for node labels.
    - title_font_size (int): Font size for subplot titles.
    - edge_label_font_size (int): Font size for edge labels.
    - figsize_per_row (int): Figure size per row for the subplot grid.
    - max_figures (int): Maximum number of figures to generate.
    - color_attribute (str): Node attribute used for coloring nodes.
    """
    
    # Extract unique values for the color attribute
    unique_values = set(nx.get_node_attributes(G, color_attribute).values())
    
    # Create a color map for the unique values
    value_to_color = {value: plt.cm.get_cmap('hsv')(i / len(unique_values))
                        for i, value in enumerate(unique_values)}
    
    # Sort nodes by degree (highest to lowest)
    sorted_nodes = sorted(G.nodes(), key=lambda node: G.degree(node), reverse=True)
    
    # Pagination settings
    total_pages = (len(sorted_nodes) + nodes_per_page - 1) // nodes_per_page
    total_pages = min(total_pages, max_figures)
    
    # List to store the total sum of edge weights for each subgraph
    edge_weight_sums = []
    
    for page in range(total_pages):
        start_idx = page * nodes_per_page
        end_idx = min(start_idx + nodes_per_page, len(sorted_nodes))
        nodes_to_plot = sorted_nodes[start_idx:end_idx]
        
        n = len(nodes_to_plot)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, figsize_per_row * rows))
        axes = axes.flatten()

        # Plot each node with its neighborhood
        for i, node in enumerate(nodes_to_plot):
            # Extract neighborhood
            subgraph_nodes = nx.single_source_shortest_path_length(G, node, cutoff=num_hops).keys()
            subgraph = G.subgraph(subgraph_nodes)

            # Set subgraph node colors and shapes
            sub_node_colors = [value_to_color.get(subgraph.nodes[n][color_attribute], 'grey') for n in subgraph.nodes()]
            sub_node_shapes = ['o' if subgraph.nodes[n]['node_type'] == 'account' else '^' for n in subgraph.nodes()]

            pos = nx.spring_layout(subgraph)

            # Draw subgraph
            ax = axes[i]
            nx.draw(subgraph, pos, ax=ax, node_color=sub_node_colors, node_size=node_size, with_labels=True, font_size=node_font_size)
            
            # Draw account nodes (circles)
            nx.draw_networkx_nodes(subgraph, pos, 
                                    node_color=[color for n, color in zip(subgraph.nodes(), sub_node_colors) if subgraph.nodes[n]['node_type'] == 'account'],
                                    node_shape='o',
                                    nodelist=[n for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'account'],
                                    node_size=node_size,
                                    ax=ax)

            # Draw customer nodes (triangles)
            nx.draw_networkx_nodes(subgraph, pos, 
                                    node_color=[color for n, color in zip(subgraph.nodes(), sub_node_colors) if subgraph.nodes[n]['node_type'] == 'customer'],
                                    node_shape='^',
                                    nodelist=[n for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'customer'],
                                    node_size=node_size,
                                    ax=ax)

            edge_weights = nx.get_edge_attributes(subgraph, 'weight')
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_weights, ax=ax, font_size=edge_label_font_size)

            # Calculate total sum of edge weights
            total_edge_weight = sum(edge_weights.values())
            edge_weight_sums.append(total_edge_weight)

            ax.set_title(f"Neighborhood (Hops: {num_hops}) of Node {node}\nTotal Edge Weight: {total_edge_weight}", fontsize=title_font_size)
            ax.axis('off')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    
    # Plot distribution of total edge weights
    plt.figure(figsize=(14, 12))
    plt.hist(edge_weight_sums, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of Total Edge Weights per Subgraph', fontsize=12)
    plt.xlabel('Total Edge Weight', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True)
    plt.show()
