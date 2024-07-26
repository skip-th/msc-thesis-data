import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import pandas as pd
import sys
from io import BytesIO
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rc('text', usetex=True)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


###############################################################################
# Define instruction types and sub-types.                                     #
###############################################################################

# All instruction types
other_instructions = ['NOP','CS2R','S2R','B2R','BAR','R2B','VOTE','TMML','TXD','SGXT','AL2P','CSMTEST','DEPBAR','IPA','ISBERD','LEPC','OUT','PIXLD','PLOP3','SETCTAID','SETLMEMBASE']
control_instructions = ['BRA','BRX','JMP','JMX','SSY','SYNC','BSYNC','WARPSYNC','CAL','CALL','JCAL','PRET','RET','BRK','PBK','CONT','PCNT','EXIT','PEXIT','BPT','BMOV','YIELD','RTT','KILL','RPCMOV','IDE','PMTRIG','BREAK','BSSY','NANOSLEEP','NANOTRAP']
memory_instructions = ['ULDC', 'LD','LDC','LDG','LDL','LDS','ST','STG','STL','STS','ATOM','ATOMS','ATOMG','RED','CCTL','MEMBAR','ERRBAR','CCTLT','CCTLL','MATCH']
conversion_instructions = ['MOV','PRMT','SEL','SHFL','CSET','CSETP','PSET','PSETP','P2R','R2P','GETLMEMBASE', 'I2F', 'F2I', 'I2I', 'F2F', 'R2UR']
integer_instructions = ['BFE','BFI','FLO','IADD','IADD3', 'UIADD','UIADD3','ICMP','IMAD','IMADSP','IMNMX','IMUL','ISCADD','ISET','ISETP','LEA','LOP3','LOP','POPC','SHF','SHL','SHR','XMAD','VABSDIFF','VABSDIFF4','BREV','IABS','IDP','QSPC','BMSK']
float_instructions = ['FADD','FCHK','FCMP','FFMA','FMNMX','FMUL','FSET','FSETP','FSWZADD','MUFU','RRO','DADD','DFMA','DMNMX','DMUL','DSET','DSETP','HADD2','HFMA2','HMMA','HMUL2','HSETP2','HSET2','FSEL']
# Memory instruction sub-types
global_memory_instructions = ['LDG','STG','LD','ST']
local_memory_instructions = ['LDL','STL']
constant_memory_instructions = ['ULDC','LDC']
shared_memory_instructions = ['LDS','STS']
atomic_instructions = ['ATOM','ATOMS','ATOMG','RED']
cache_control_instructions = ['CCTL','CCTLT','CCTLL']
sync_and_error_instructions = ['MEMBAR','ERRBAR','MATCH']


###############################################################################
# Parse SASS file.                                                            #
###############################################################################

functions_indices = []
lines = []
auto_function_exploration = True

if len(sys.argv) < 2:
    raise Exception('Error: No path to SASS file provided.')
path = sys.argv[1]

with open(path, 'r') as f:
    line_function_start = -1
    line_function_end = -1
    line_idx = 0
    function_header = True

    for i,line in enumerate(f.readlines()):
        line_idx = i
        lines.append(line)

        if line.startswith('Kernel Name'): # header / start of function
            line_function_start = line_idx
            function_header = True

        elif line.startswith('------------------') and function_header: # end of header
            function_header = False

        elif line.startswith('------------------'): # multiple functions in single file
            line_function_end = line_idx
            functions_indices.append((line_function_start, line_function_end))

    line_function_end = line_idx # last function in file
    functions_indices.append((line_function_start, line_function_end)) # watch out, errors already at start of file


###############################################################################
# Split functions into kernel names, parameters, SASS sources, and addresses. #
###############################################################################
functions = dict()
for i, (start, end) in enumerate(functions_indices):
    name = lines[start].replace('Kernel Name','').strip().split('(')[0]
    parameters = lines[start].replace('Kernel Name','').strip().split('(')[1].split(')')[0] if '(' in lines[start] else ''

    sass_start = -1
    for j, line in enumerate(lines[start:end]):
        if 'Address' in line:
            sass_start = start + j + 1
            break
    sass_lines = lines[sass_start:end]

    addresses = []
    sources = []
    for line in sass_lines:
        if '0x' in line:
            address = line.strip().split(' ',1)[0].strip()
            source = line.strip().split(' ',1)[1].strip()

            if source == 'NOP':
                break
            addresses.append(address)
            sources.append(source)

    functions[name] = {
        'parameters' : parameters,
        'address' : addresses,
        'source raw' : sources
    }


###############################################################################
# Split sources into instructions and operands.                               #
###############################################################################

def remvove_duplicate_whitespace(string):
    return ' '.join(string.split())

for name, data in functions.items():
    raw_source = data['source raw']
    source = []

    for line in raw_source:
        line = remvove_duplicate_whitespace(line)
        if not line.startswith('@'):
            conditional, instruction, *operands = '', *line.split(' ')
            instruction, *modifiers = instruction.split('.')
        elif len(line.split(' ')) >= 3:
            conditional, instruction, *operands = line.split(' ')
            instruction, *modifiers = instruction.split('.')
        elif len(line.split(' ')) == 2:
            conditional, instruction, operands = (*line.split(' '), [])
            instruction, *modifiers = instruction.split('.')
        source.append([conditional, instruction, modifiers, operands])

    functions[name]['source'] = source

# Convert to data frame
functions_dfs = dict()
for function_name, data in functions.items():
    parameters = data['parameters']
    addresses = data['address']
    source = data['source']
    raw = data['source raw']

    array_2d = np.array(source, dtype=object)
    for i in range(array_2d.shape[0]):
        if len(array_2d[i]) == 1:
            array_2d[i] = np.append(array_2d[i], None)  # Append None (or any other padding value)
    padded_array = np.zeros((array_2d.shape[0], 4), dtype=object)
    padded_array[:, 0] = [row[0] for row in array_2d]
    padded_array[:, 1] = [row[1] for row in array_2d]
    padded_array[:, 2] = [row[2] for row in array_2d]
    rm_comma = lambda x: [y.replace(',','') for y in x]
    padded_array[:, 3] = [rm_comma(row[3]) for row in array_2d]

    functions_dfs[function_name] = pd.DataFrame(addresses, columns=['address'])
    functions_dfs[function_name]['conditional'] = padded_array[:, 0]
    functions_dfs[function_name]['instruction'] = padded_array[:, 1]
    functions_dfs[function_name]['modifiers']   = padded_array[:, 2]
    functions_dfs[function_name]['operands']    = padded_array[:, 3]


###############################################################################
# Explore special control flow instructions.                                  #
###############################################################################

def explore_function(df, walked_df, function_name, function_names, ret_addr, auto=False):
    if auto and len(function_names) > 2:
        raise Exception("Error: Cannot automatically determine called function with more than two functions. ({})".format(function_names))

    function_names = list(reversed(df.keys()))
    data = df[function_name]
    print(function_name)

    x = 0
    for row in data.iterrows():
        idx, column_data = row

        address, conditional, instruction, modifiers, operands = column_data
        print(address, conditional, instruction + '.' + '.'.join(modifiers), *operands)
        column_data['recursive'] = False

        if instruction == 'CONT' or instruction == 'BRK':
            raise Exception('Error: Loop control instructions (i.e., CONT and BRK) not implemented.')

        if instruction == 'CALL': # call to another function
            if auto and len(function_names) == 2:
                called_function_name = function_names[1]
            elif auto and len(function_names) == 1:
                called_function_name = function_names[0]
            else:
                print('##########################')
                print('Encountered function call in function: \'{}\''.format(function_name))
                print(address, conditional, instruction + '.' + '.'.join(modifiers), *operands)
                print('Functions:\n\t{}'.format('\n\t'.join([f'{i}: {j}' for i,j in enumerate(function_names)])))
                call_dest = int(input('Select function:'))
                x += 1
                called_function_name = function_names[call_dest]
                print('##########################')

            column_data['operands'] = [df[called_function_name]['address'].loc[0]]

            if called_function_name == function_name: # recursive function call
                column_data['recursive'] = True

            walked_df.loc[len(walked_df)] = column_data

            if called_function_name != function_name: # no recursive function call
                ret_addr = data['address'].loc[idx + 1]
                walked_df = explore_function(df, walked_df, called_function_name, function_names, ret_addr, auto)

        elif instruction == 'RET':
            column_data['operands'][1] = ret_addr
            walked_df.loc[len(walked_df)] = column_data
        else:
            walked_df.loc[len(walked_df)] = column_data
        if conditional == '' and (instruction == 'EXIT' or instruction == 'RET'):
            return walked_df.copy()

    return walked_df.copy()

function_names = list(reversed(functions_dfs.keys()))
walked_df = pd.DataFrame(columns=np.concatenate((functions_dfs[function_names[0]].columns, ['recursive'])))
walked_df['recursive'] = False
if auto_function_exploration:
    function_name = function_names[0]
    starting_point_input = 0
else:
    print('Functions:\n\t{}'.format('\n\t'.join([f'{i}: {j}' for i,j in enumerate(function_names)])))
    starting_point = int(input('Select entry function:'))
    starting_point_input = starting_point
    print('##########################')
    function_name = function_names[starting_point]

start_addr = functions_dfs[function_name]['address'].loc[0]
walked_df = explore_function(functions_dfs, walked_df, function_name, function_names, start_addr, auto=auto_function_exploration)

def add_list_element(df, index, data):
    if df.loc[index] != None:
        df.loc[index].append(data)
    else:
        df.loc[index] = [data]

def find_dest_addr(df, curr_idx, address_src, address_dst):
    index_dest = -1
    if int(address_src, 16) < int(address_dst, 16): # if the destination is a higher address
        for i in range(curr_idx, len(df)):
            if df.loc[i] == address_dst:
                index_dest = i
                break
    elif int(address_src, 16) > int(address_dst, 16): # if the destination is a lower address
        for i in reversed(range(curr_idx)):
            if df.loc[i] == address_dst:
                index_dest = i
                break
    if index_dest == -1:
        raise Exception('Failed to find address {} from address {}'.format(address_dst, address_src))
    return index_dest

walked_df['destination'] = None
last_idx = len(walked_df['destination'])
prev_addrs = []
for row in walked_df.iterrows():
    index_src, column_data = row
    address, conditional, instruction, modifiers, operands, recursive, _ = column_data
    prev_addrs.append(address)

    if conditional != '' and (instruction == 'BRA' or instruction == 'BRX'): # If there is a conditional branch
        index_dest = find_dest_addr(walked_df['address'], index_src, address, operands[0])
        add_list_element(walked_df['destination'], index_src, index_dest) # add branch address as neighbor vertex
        add_list_element(walked_df['destination'], index_src, index_src + 1 if index_src + 1 != last_idx else 'END') # add next instruction as another neighbor vertex
    elif conditional == '' and (instruction == 'BRA' or instruction == 'BRX' or instruction == 'JMP'): # If there is an unconditional branch
        index_dest = find_dest_addr(walked_df['address'], index_src, address, operands[0])
        add_list_element(walked_df['destination'], index_src, index_dest) # add branch address as sole neighbor vertex
    elif instruction == 'CALL' and recursive == False: # If there is a conditional branch
        add_list_element(walked_df['destination'], index_src, index_src + 1) # add next instruction as another neighbor vertex
    elif instruction == 'CALL' and recursive == True:
        add_list_element(walked_df['destination'], index_src, index_src + 1) # add next instruction as another neighbor vertex
        index_dest = -1
        for i in reversed(range(index_src)):
            if walked_df['instruction'].loc[i] == 'CALL':
                index_dest = i + 1
                break
        add_list_element(walked_df['destination'], index_src, index_dest) # add branch address as neighbor vertex
    elif conditional != '' and instruction == 'EXIT': # If there is a conditional exit
        add_list_element(walked_df['destination'], index_src, 'END') # add branch address as neighbor vertex
        add_list_element(walked_df['destination'], index_src, index_src + 1 if index_src + 1 != last_idx else 'END') # add next instruction as another neighbor vertex
    elif conditional == '' and instruction == 'EXIT': # If there is a unconditional exit
        add_list_element(walked_df['destination'], index_src, 'END') # add branch address as sole neighbor vertex
    elif instruction == 'RET': # If there is a conditional return
        add_list_element(walked_df['destination'], index_src, index_src + 1) # add next instruction as another neighbor vertex
    else: # if there are no control instructions
        add_list_element(walked_df['destination'], index_src, index_src + 1 if index_src + 1 != last_idx else 'END') # add next instruction as neighbor vertex


###############################################################################
# Visualize the execution paths as a graph.                                   #
###############################################################################

def visualize_graph(names, sources, destinations, savefig='', layout=0):
    import networkx as nx
    from networkx.drawing.nx_agraph import to_agraph

    G = nx.DiGraph()
    for source, name in zip(sources, names):
        G.add_node(source, label=name)
    for source, dest_list in zip(sources, destinations):
        for dest in dest_list:
            G.add_edge(source, dest)

    A = to_agraph(G)  # Convert to a PyGraphviz graph
    if layout == 0:
        A.graph_attr['splines'] = 'ortho'  # Use orthogonal edges, cannot use together with box shape!
        A.layout(prog='dot')  # Use 'dot' or 'neato' with orthogonal edge support
    elif layout == 1:
        A.graph_attr['splines'] = 'polyline'  # Use orthogonal edges, cannot use together with box shape!
        A.node_attr['shape'] = 'box'  # Optional: make nodes box-shaped
        A.layout(prog='dot')  # Use 'dot' or 'neato' with orthogonal edge support
    elif layout >= 2:
        A.graph_attr['splines'] = 'ortho'  # Use orthogonal edges, cannot use together with box shape!
        A.layout(prog='neato')  # Use 'dot' or 'neato' with orthogonal edge support

    # Save as PDF if filename provided
    if savefig:
        A.draw(savefig + '.pdf')

def memory_subtype_colors(name):
    # transition between yellow/gray -> yellow -> yellow/orange
    if name in sync_and_error_instructions:
        color = '#BD9D8C'
    elif name in cache_control_instructions:
        color = '#D5AE74'
    elif name in shared_memory_instructions:
        color = '#EDBF5B'
    elif name in constant_memory_instructions:
        color = '#F9BF4F'
    elif name in local_memory_instructions:
        color = '#F9AE4F'
    elif name in global_memory_instructions:
        color = '#F99D4F'
    elif name in atomic_instructions:
        color = '#FA8C50'
    else:
        raise Exception('Error: Unknown memory subtype instruction:', name)
    return color

def add_color(name, special, use_subtypes=False):
    if name in special:
        color = special[name]
    else:
        if name in float_instructions:
            color = '#219ebc' # blue
        elif name in integer_instructions:
            color = '#8ecae6' # lightblue
        elif name in conversion_instructions:
            color = '#43aa8b' # green-blue
        elif name in memory_instructions:
            color = '#f9c74f' # yellow
            if use_subtypes:
                color = memory_subtype_colors(name)
        elif name in control_instructions:
            color = '#b7b7a4' # gray
        elif name in other_instructions:
            color = '#cb997e' # brown
        else:
            raise Exception('Error: Unknown instruction:', name)

    return color

def visualize_graph_color(names, sources, destinations, savefig='', layout=0, special=dict(), use_subtypes=False):
    import networkx as nx
    from networkx.drawing.nx_agraph import to_agraph
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    # Adding nodes with labels and specific colors if applicable
    for source, name in zip(sources, names):
        color = add_color(name, special, use_subtypes)
        G.add_node(source, label=name, fillcolor=color, style='filled')

    # Adding edges
    for source, dest_list in zip(sources, destinations):
        for dest in dest_list:
            G.add_edge(source, dest)

    A = to_agraph(G)  # Convert to a PyGraphviz graph

    # Setting layout options based on layout parameter
    if layout == 0:
        A.graph_attr['splines'] = 'ortho'
        A.layout(prog='dot')
    elif layout == 1:
        A.graph_attr['splines'] = 'polyline'
        A.node_attr['shape'] = 'box'
        A.layout(prog='dot')
    elif layout >= 2:
        A.graph_attr['splines'] = 'ortho'
        A.layout(prog='neato')

    # Save as PDF if filename provided
    if savefig:
        A.draw(savefig + '_color.pdf')

# Visualize regular graph
names = ['START'] + list(walked_df['instruction'])
self_idx = ['-1'] + list(walked_df.index)
neighbors = ['0'] + list(walked_df['destination'])
visualize_graph(names, self_idx, neighbors, function_name, layout=0)

# Visualize colored graph
names = ['START'] + list(walked_df['instruction']) + ['END']
self_idx = ['-1'] + list(walked_df.index) + ['END']
neighbors = ['0'] + list(walked_df['destination']) + []
special_colors = { # Add start and end nodes with specific colors
    'START': '#90be6d', # light green
    'END': '#FA5250'} # light red
visualize_graph_color(names, self_idx, neighbors, function_name, layout=1, special=special_colors, use_subtypes=True)


###############################################################################
# Identify instructions with variable workload.                               #
###############################################################################

# Find branches in walked_df that go to a previous address, indicating a loop.
def find_loops(df):
    loops = []
    for row in df.iterrows():
        index_src, column_data = row
        instruction = column_data[2]
        if len(column_data['destination']) == 2 and instruction == 'BRA' or instruction == 'BRX' or instruction == 'JMP':
            for dest in column_data['destination']:
                if dest < index_src:
                    loops.append((index_src, dest))
    return loops

loops = find_loops(walked_df)
print('Found {} backward branches (loops): {}'.format(len(loops), loops))

# Find recursive calls in walked_df include destination index
def find_recursion_dest(df):
    recursions = []
    for row in df.iterrows():
        index_src, column_data = row
        recursive = column_data[5]
        if recursive and len(column_data['destination']) == 2:
            recursions.append((index_src, column_data['destination'][1]))
    return recursions

recursive_calls = find_recursion_dest(walked_df)
print('Found {} conditional recursive calls: {}'.format(len(recursive_calls), recursive_calls))

# Find branches that skip forward
def find_skips(df):
    skips = []
    for row in df.iterrows():
        index_src, column_data = row
        _,_,instruction,_,_,_,destination = column_data
        if instruction == 'BRA' or instruction == 'BRX' or instruction == 'JMP':
            if len(destination) == 2 and destination[0] > index_src and destination[1] > index_src:
                skips.append((index_src, destination[0] if destination[0] > destination[1] else destination[1]))
    return skips

skips = find_skips(walked_df)
print('Found {} forward branches: {}'.format(len(skips), skips))


###############################################################################
# Visualize the execution paths as a graph with highlighted loops.            #
###############################################################################

def visualize_graph_clusters(names, sources, destinations, savefig='', layout=0, highlight_ranges=None, use_clusters=True, cluster_name='cluster', use_subtypes=False, special=dict(), show=True, save=True):
    G = nx.DiGraph()
    # Adding all nodes and edges first
    for source, name in zip(sources, names):
        G.add_node(source, label=name)
    for source, dest_list in zip(sources, destinations):
        for dest in dest_list:
            G.add_edge(source, dest)

    # Convert to PyGraphviz graph
    A = to_agraph(G)

    # Adding nodes with labels and specific colors if applicable
    for source, name in zip(sources, names):
        color = add_color(name, special, use_subtypes)
        if not use_clusters:
            # A.add_node(source, label=name, fillcolor=color, style='filled, bold')
            if highlight_ranges and isinstance(source, int) and any(int(source) in range(start, end+1) for start, end in highlight_ranges):
                # Customize the appearance for highlighted nodes
                A.add_node(source, label=name, fillcolor=color, style='filled, bold')
            else:
                A.add_node(source, label=name, fillcolor='white', style='filled')
        else:
            A.add_node(source, label=name, fillcolor=color, style='filled, bold')


    # Create clusters for the highlighted ranges
    loop_names = []
    if highlight_ranges:
        # nested_depth = 0
        upper_loop = 0
        j = 0
        for i, (start, end) in enumerate(highlight_ranges):
            # Correct method to create and modify subgraphs (clusters)
            if use_clusters:
                c = A.add_subgraph(name=f'cluster_{i}', rank='same')
            # Check if loop is nested in previous loop
            if i > 0 and start > highlight_ranges[i-1][0] and end < highlight_ranges[i-1][1]:
                j += 1
                upper_loop = str(upper_loop) + '.' + str(j)
                if use_clusters:
                    c.graph_attr['label'] = f'Nested {cluster_name} {upper_loop}'
            else:
                j = 0
                upper_loop = str(i + 1)
                if use_clusters:
                    c.graph_attr['label'] = f'{cluster_name.capitalize()} {upper_loop}'
            loop_names.append(upper_loop)
            if use_clusters:
                c.graph_attr['color'] = 'blue'
                c.graph_attr['style'] = 'bold'
                for source, name in zip(sources, names):
                    if isinstance(source, int) and int(source) in range(start, end+1):
                        c.add_node(source, fillcolor=add_color(name, special, use_subtypes), style='filled, bold')

    # Setting layout configurations
    if layout == 0:
        A.graph_attr['splines'] = 'ortho'
        A.layout(prog='dot')
    elif layout == 1:
        A.graph_attr['splines'] = 'polyline'
        A.node_attr['shape'] = 'box'
        A.layout(prog='dot')
    elif layout >= 2:
        A.graph_attr['splines'] = 'ortho'
        A.layout(prog='neato')

    # Save and display the graph
    if save:
        A.draw(savefig + '_highlighted_' + cluster_name + '_' + str(highlight_ranges) + '.pdf')

    return loop_names


# Visualize colored graph
names = ['START'] + list(walked_df['instruction']) + ['END']
self_idx = ['-1'] + list(walked_df.index) + ['END']
neighbors = ['0'] + list(walked_df['destination']) + []
special_colors = { # Add start and end nodes with specific colors
'START': '#90be6d', # light green
'END': '#FA5250'} # light red

# Highlight loops
print('Highlighting loops')
loops_sorted = [(end, start) if start > end else (start, end) for start, end in loops] # invert ranges to ensure that smallest number is first
loops_sorted.sort(key=lambda x: x[0])
named_loops = visualize_graph_clusters(names, self_idx, neighbors, function_name, layout=1, highlight_ranges=loops_sorted, cluster_name='loop', special=special_colors, use_subtypes=True, show=True, save=True)

# Highlight recursive calls
print('Highlighting recursive calls')
recursive_calls_sorted = [(end, start) if start > end else (start, end) for start, end in recursive_calls] # invert ranges to ensure that smallest number is first
recursive_calls_sorted.sort(key=lambda x: x[0])
named_recursive_calls = visualize_graph_clusters(names, self_idx, neighbors, function_name, layout=1, highlight_ranges=recursive_calls_sorted, cluster_name='recursive call', special=special_colors, use_subtypes=True, show=True, save=True)

# # Individually highlight forward branches (skip instructions)
# print('Highlighting conditional skips (if-statements?)')
# skips_sorted = [(end, start) if start > end else (start, end) for start, end in skips]
# skips_sorted.sort(key=lambda x: x[0])
# named_skips = visualize_graph_clusters(names, self_idx, neighbors, function_name, layout=1, highlight_ranges=skips_sorted, cluster_name='skip', use_clusters=True, special=special_colors, use_subtypes=True, show=False, save=False)
# for highlight_range in skips_sorted:
#     visualize_graph_clusters(names, self_idx, neighbors, function_name, layout=1, highlight_ranges=[highlight_range], cluster_name='skip', use_clusters=False, special=special_colors, use_subtypes=True, show=True, save=True)
