import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
import datetime
from plotly.subplots import make_subplots

def create_cyberpunk_theme():
    """Create a cyberpunk-themed template for plots."""
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#ff00ff', 'family': 'Inter'},
        'scene': {
            'xaxis': {
                'gridcolor': 'rgba(0,255,255,0.2)',
                'showbackground': True,
                'backgroundcolor': 'rgba(10,10,30,0.8)',
                'showgrid': True,
                'gridwidth': 2,
                'title': {'text': '', 'font': {'color': '#00ffff'}},
                'linecolor': '#00ffff',
                'linewidth': 3
            },
            'yaxis': {
                'gridcolor': 'rgba(255,0,255,0.2)',
                'showbackground': True,
                'backgroundcolor': 'rgba(10,10,30,0.8)',
                'showgrid': True,
                'gridwidth': 2,
                'title': {'text': '', 'font': {'color': '#ff00ff'}},
                'linecolor': '#ff00ff',
                'linewidth': 3
            },
            'zaxis': {
                'gridcolor': 'rgba(255,255,0,0.2)',
                'showbackground': True,
                'backgroundcolor': 'rgba(10,10,30,0.8)',
                'showgrid': True,
                'gridwidth': 2,
                'title': {'text': '', 'font': {'color': '#ffff00'}},
                'linecolor': '#ffff00',
                'linewidth': 3
            },
            'camera': {
                'eye': {'x': 1.8, 'y': 1.8, 'z': 1.8},
                'projection': {'type': 'perspective'}
            },
            'aspectratio': {'x': 1, 'y': 1, 'z': 0.8}
        },
        'margin': {'l': 0, 'r': 0, 't': 30, 'b': 0}
    }

def create_entropy_plot(entropy_value, lower_staging=True):
    """Create a cyberpunk 3D entropy visualization with adjustable positioning."""
    # Generate more complex patterns for visualization
    t = np.linspace(0, 16*np.pi, 2000)
    scale = entropy_value / 8  # Normalize to max entropy of 8
    
    # Create expanding double spiral with modulation
    x1 = (t/8 * np.cos(t) + 0.3*np.sin(3*t)) * scale
    y1 = (t/8 * np.sin(t) + 0.3*np.cos(3*t)) * scale
    z1 = (t/10 + 0.2*np.sin(5*t)) * scale
    
    # Second spiral path (offset)
    x2 = (t/8 * np.cos(t + np.pi) + 0.3*np.sin(3*t)) * scale
    y2 = (t/8 * np.sin(t + np.pi) + 0.3*np.cos(3*t)) * scale
    z2 = (t/10 + 0.2*np.sin(5*t + np.pi)) * scale
    
    # Create holographic cube effect
    cube_size = scale * 1.5
    edges_x = []
    edges_y = []
    edges_z = []
    
    # Generate cube corners
    corners = [
        [-cube_size, -cube_size, -cube_size],
        [cube_size, -cube_size, -cube_size],
        [cube_size, cube_size, -cube_size],
        [-cube_size, cube_size, -cube_size],
        [-cube_size, -cube_size, cube_size],
        [cube_size, -cube_size, cube_size],
        [cube_size, cube_size, cube_size],
        [-cube_size, cube_size, cube_size]
    ]
    
    # Connect cube edges (12 edges total)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    for edge in edges:
        c1, c2 = corners[edge[0]], corners[edge[1]]
        # Create points along the edge
        for i in range(20):
            t = i / 19.0
            edges_x.append(c1[0] * (1-t) + c2[0] * t)
            edges_y.append(c1[1] * (1-t) + c2[1] * t)
            edges_z.append(c1[2] * (1-t) + c2[2] * t)
    
    # Create voxel-like points inside the cube
    voxel_points = 800
    voxel_x = np.random.uniform(-cube_size, cube_size, voxel_points)
    voxel_y = np.random.uniform(-cube_size, cube_size, voxel_points)
    voxel_z = np.random.uniform(-cube_size, cube_size, voxel_points)
    voxel_colors = np.sqrt(
        (voxel_x/cube_size)**2 + 
        (voxel_y/cube_size)**2 + 
        (voxel_z/cube_size)**2
    )
    
    # Create the pulsating sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = scale * 0.8 * np.outer(np.cos(u), np.sin(v))
    sphere_y = scale * 0.8 * np.outer(np.sin(u), np.sin(v))
    sphere_z = scale * 0.8 * np.outer(np.ones(50), np.cos(v))
    
    # Create ripple effect on sphere
    for i in range(len(u)):
        for j in range(len(v)):
            ripple = 0.1 * np.sin(8 * u[i]) * np.sin(8 * v[j])
            sphere_x[i, j] += ripple * np.cos(u[i]) * np.sin(v[j])
            sphere_y[i, j] += ripple * np.sin(u[i]) * np.sin(v[j])
            sphere_z[i, j] += ripple * np.cos(v[j])
    
    # Create circular rings around center
    rings = []
    for radius in np.linspace(0.2, 1.0, 5):
        ring_t = np.linspace(0, 2*np.pi, 200)
        ring_x = radius * scale * np.cos(ring_t)
        ring_y = radius * scale * np.sin(ring_t)
        ring_z = np.zeros_like(ring_t) + (radius * 0.1 * scale)
        rings.append((ring_x, ring_y, ring_z))
    
    # Combine all visualizations into figure
    fig = go.Figure()
    
    # 1. Add core sphere with ripple effect
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        colorscale=[
            [0, '#00ffff'], 
            [0.5, '#ff00ff'],
            [1, '#ffff00']
        ],
        opacity=0.6,
        showscale=False,
        hoverinfo='none',
        name='Entropy Core'
    ))
    
    # 2. Add primary data spiral
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines',
        line=dict(
            color='#ff00ff',
            width=4
        ),
        hoverinfo='none',
        name='Data Flow'
    ))
    
    # 3. Add secondary data spiral
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines',
        line=dict(
            color='#00ffff',
            width=4
        ),
        hoverinfo='none',
        name='Mirror Flow'
    ))
    
    # 4. Add data points along spirals
    fig.add_trace(go.Scatter3d(
        x=x1[::100], y=y1[::100], z=z1[::100],
        mode='markers',
        marker=dict(
            size=6,
            color=z1[::100],
            colorscale=[[0, '#ff00ff'], [1, '#00ffff']],
            opacity=0.8,
            symbol='circle'
        ),
        hoverinfo='none',
        name='Energy Nodes'
    ))
    
    # 5. Add cube framework
    fig.add_trace(go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='markers',
        marker=dict(
            size=3,
            color='#ffff00',
            opacity=0.7
        ),
        hoverinfo='none',
        name='Data Boundary'
    ))
    
    # 6. Add voxel points inside the cube
    fig.add_trace(go.Scatter3d(
        x=voxel_x, y=voxel_y, z=voxel_z,
        mode='markers',
        marker=dict(
            size=2,
            color=voxel_colors,
            colorscale=[[0, '#00ffff'], [0.5, '#ff00ff'], [1, '#ffff00']],
            opacity=0.3
        ),
        hoverinfo='none',
        name='Data Cloud'
    ))
    
    # 7. Add circular rings
    for i, (ring_x, ring_y, ring_z) in enumerate(rings):
        fig.add_trace(go.Scatter3d(
            x=ring_x, y=ring_y, z=ring_z,
            mode='lines',
            line=dict(
                color='#ffff00' if i % 2 == 0 else '#00ffff',
                width=3
            ),
            opacity=0.7,
            hoverinfo='none',
            name=f'Data Ring {i+1}'
        ))
    
    # 8. Add entropy value indicator (just the marker)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[scale*1.8],
        mode='markers',
        marker=dict(
            size=16,
            color='#ff00ff',
            symbol='diamond',
            line=dict(color='#00ffff', width=2)
        ),
        name='Entropy Value'
    ))
    
    # Add entropy text separate from the marker, positioned to the side
    fig.add_trace(go.Scatter3d(
        x=[scale*1.2], y=[scale*0.5], z=[scale*1.8],
        mode='text',
        text=[f'Entropy: {entropy_value:.4f}'],
        textfont=dict(
            color='#ffff00',
            size=14
        ),
        name='Entropy Value Text'
    ))
    
    # Add small holographic info panels
    panel_offsets = [
        [cube_size*1.2, 0, 0],
        [-cube_size*1.2, 0, 0],
        [0, cube_size*1.2, 0]
    ]
    
    panel_texts = [
        f"ENTROPY: {entropy_value:.4f}",
        f"NORM: {entropy_value/8:.2f}",
        f"STATUS: {'HIGH' if entropy_value > 7 else 'NORMAL' if entropy_value > 5 else 'LOW'}"
    ]
    
    for i, (offset, text) in enumerate(zip(panel_offsets, panel_texts)):
        fig.add_trace(go.Scatter3d(
            x=[offset[0]], y=[offset[1]], z=[offset[2]],
            mode='text',
            text=[text],
            textfont=dict(
                color='#00ffff' if i == 0 else '#ff00ff' if i == 1 else '#ffff00',
                size=12
            ),
            name=f'Info Panel {i+1}'
        ))
    
    # Layout configuration
    fig.update_layout(
        title={
            'text': 'ENTROPY VISUALIZATION MATRIX',
            'font': {'color': '#00ffff', 'size': 24}
        },
        showlegend=True,
        legend=dict(
            font=dict(color='#00ffff', size=10),
            bgcolor='rgba(0,0,10,0.7)',
            bordercolor='#ff00ff',
            borderwidth=1
        ),
        **create_cyberpunk_theme()
    )
    
    # Add 2D annotation with entropy value at the bottom for better visibility
    fig.add_annotation(
        x=0.5,  # Centered horizontally
        y=0.02,  # Very bottom
        text=f"<b>ENTROPY VALUE:</b> {entropy_value:.6f} | <b>NORMALIZED:</b> {entropy_value/8:.4f} | <b>STATUS:</b> {'HIGH' if entropy_value > 7 else 'NORMAL' if entropy_value > 5 else 'LOW'}",
        showarrow=False,
        font=dict(
            family="monospace",
            size=14,
            color="#00ffff"
        ),
        align="center",
        bgcolor="rgba(0,0,30,0.7)",
        bordercolor="#ff00ff",
        borderwidth=2,
        borderpad=6,
        xref="paper",
        yref="paper"
    )
    
    # Position the graph lower to prevent obstruction
    if lower_staging:
        # Position the graph lower in the visualization container
        fig.update_layout(
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.3),  # Move center down
                eye=dict(x=1.8, y=1.8, z=1.2)   # Look from higher angle
            ),
            # Add more margin at the top to prevent obstruction
            margin=dict(t=80, b=20, l=20, r=20)
        )
    else:
        # Standard camera settings if not using lower staging
        fig.update_layout(
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.8, y=1.8, z=0.8)
            )
        )
    
    return fig

def create_byte_frequency_plot(bytes_values, frequencies, lower_staging=True):
    """Create a cyberpunk 3D byte frequency visualization with adjustable positioning."""
    # Normalize frequencies
    freq_norm = np.array(frequencies) / max(frequencies)
    x = np.array(bytes_values)
    
    # Create heightmap data for primary visualization
    X, Y = np.meshgrid(
        np.linspace(0, 255, 100),  # Use full byte range (0-255)
        np.linspace(0, 1, 100)
    )
    
    # Create complex terrain with multiple peaks
    Z = np.zeros_like(X)
    for i, (bv, fn) in enumerate(zip(x, freq_norm)):
        # Use more complex function to create sharper peaks
        Z += fn * np.exp(-0.005 * ((X - bv)**2 + (Y - fn*5)**2))
    
    # Amplify the terrain for more dramatic effect
    Z = Z * 1.5
    
    # Create city-like grid layout for the visualization
    city_x = []
    city_y = []
    city_z = []
    city_colors = []
    
    # Generate vertical bars ("buildings") based on frequency data
    for i, (bv, fn) in enumerate(zip(x, freq_norm)):
        if fn > 0.01:  # Only plot significant frequencies
            # Create building with height proportional to frequency
            height = fn * 1.5
            
            # Base of building
            base_size = 0.8
            base_x = [bv-base_size, bv+base_size, bv+base_size, bv-base_size, bv-base_size]
            base_y = [0, 0, 1, 1, 0]
            base_z = [0, 0, 0, 0, 0]
            
            # Top of building
            top_x = [bv-base_size, bv+base_size, bv+base_size, bv-base_size, bv-base_size]
            top_y = [0, 0, 1, 1, 0]
            top_z = [height, height, height, height, height]
            
            # Vertical lines
            for j in range(4):
                city_x.extend([base_x[j], top_x[j], None])
                city_y.extend([base_y[j], top_y[j], None])
                city_z.extend([base_z[j], top_z[j], None])
                # Assign color based on position in spectrum
                color_val = (bv / 255)
                city_colors.extend([color_val, color_val, color_val])
    
    # Create holographic data cube (3D grid)
    grid_size = 4
    grid_x, grid_y, grid_z = [], [], []
    
    # Create 3D grid lines
    for i in range(grid_size + 1):
        # Normalized coordinates (0 to 1)
        coord = i / grid_size
        
        # Lines along X-axis
        for j in range(grid_size + 1):
            jcoord = j / grid_size
            grid_x.extend([coord, coord, None])
            grid_y.extend([jcoord, jcoord, None])
            grid_z.extend([0, 1, None])
            
            grid_x.extend([coord, coord, None])
            grid_y.extend([0, 1, None])
            grid_z.extend([jcoord, jcoord, None])
        
        # Lines along Y-axis
        for j in range(grid_size + 1):
            jcoord = j / grid_size
            grid_x.extend([0, 1, None])
            grid_y.extend([coord, coord, None])
            grid_z.extend([jcoord, jcoord, None])
    
    # Create primary visualization with multiple components
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scene'}]],
        subplot_titles=["BYTE FREQUENCY MATRIX"]
    )
    
    # 1. Main terrain surface (uses custom colorscale for cyberpunk look)
    fig.add_trace(
        go.Surface(
            x=X, 
            y=Y, 
            z=Z,
            colorscale=[
                [0, '#000000'],
                [0.2, '#00ffff'],
                [0.4, '#0000ff'],
                [0.6, '#ff00ff'],
                [0.8, '#ff0000'],
                [1, '#ffff00']
            ],
            opacity=0.8,
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.8,
                roughness=0.5,
                fresnel=0.8
            ),
            hoverinfo='none',
            name='Frequency Terrain'
        )
    )
    
    # 2. Add "city" buildings for each frequency peak
    fig.add_trace(
        go.Scatter3d(
            x=city_x, 
            y=city_y, 
            z=city_z,
            mode='lines',
            line=dict(
                color=city_colors,
                colorscale=[
                    [0, '#00ffff'],
                    [0.5, '#ff00ff'],
                    [1, '#ffff00']
                ],
                width=2
            ),
            hoverinfo='none',
            name='Byte Buildings'
        )
    )
    
    # 3. Add grid overlay
    fig.add_trace(
        go.Scatter3d(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            mode='lines',
            line=dict(
                color='#00ffff',
                width=1
            ),
            opacity=0.3,
            hoverinfo='none',
            name='Data Grid'
        )
    )
    
    # 4. Add data points
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=np.full_like(x, 0.5),  # Center Y
            z=freq_norm * 1.5,  # Scale Z 
            mode='markers',
            marker=dict(
                size=4,
                color=x,  # Color by byte value
                colorscale=[
                    [0, '#00ffff'],
                    [0.5, '#ff00ff'],
                    [1, '#ffff00']
                ],
                opacity=0.8,
                symbol='circle'
            ),
            hoverinfo='none',
            name='Data Points'
        )
    )
    
    # 5. Add connecting lines for data points (neon circuit look)
    significant_indices = freq_norm > 0.05
    if any(significant_indices):
        sig_x = x[significant_indices]
        sig_z = freq_norm[significant_indices] * 1.5
        
        # Sort by byte value to create path
        sort_idx = np.argsort(sig_x)
        sorted_x = sig_x[sort_idx]
        sorted_z = sig_z[sort_idx]
        
        fig.add_trace(
            go.Scatter3d(
                x=sorted_x,
                y=np.full_like(sorted_x, 0.5),  # Center Y
                z=sorted_z,
                mode='lines',
                line=dict(
                    color='#ff00ff',
                    width=3
                ),
                opacity=0.8,
                hoverinfo='none',
                name='Frequency Circuit'
            )
        )
    
    # 6. Add highlight points for anomalies (high frequencies)
    anomaly_threshold = 0.7
    anomaly_indices = freq_norm > anomaly_threshold
    if any(anomaly_indices):
        anomaly_x = x[anomaly_indices]
        anomaly_z = freq_norm[anomaly_indices] * 1.5
        
        fig.add_trace(
            go.Scatter3d(
                x=anomaly_x,
                y=np.full_like(anomaly_x, 0.5),
                z=anomaly_z + 0.1,  # Slightly above
                mode='markers',
                marker=dict(
                    size=8,
                    color='#ffff00',
                    symbol='diamond',
                    line=dict(
                        color='#ff00ff',
                        width=2
                    )
                ),
                hoverinfo='none',
                name='Anomalies'
            )
        )
    
    # 7. Add informational panel text
    # Calculate some statistics
    freq_avg = np.mean(freq_norm)
    freq_std = np.std(freq_norm)
    freq_max = np.max(freq_norm)
    max_byte = x[np.argmax(freq_norm)]
    
    # Information panels
    panel_texts = [
        f"MAX BYTE: {int(max_byte)} ({max_byte:02X}h)",
        f"FREQ STD: {freq_std:.4f}",
        f"ENTROPY INDEX: {freq_std/freq_avg:.2f}"
    ]
    
    # Place text panels in 3D space at a better position
    for i, text in enumerate(panel_texts):
        fig.add_trace(
            go.Scatter3d(
                x=[255 * 1.3],  # Further right side
                y=[0.2 + i*0.2],  # Staggered Y positions
                z=[0.8],         # Fixed height for better visibility
                mode='text',
                text=[text],
                textfont=dict(
                    color='#00ffff' if i == 0 else '#ff00ff' if i == 1 else '#ffff00',
                    size=12
                ),
                hoverinfo='none',
                name=f'Info {i+1}'
            )
        )
    
    # 8. Add background glowing effect with small particles
    particles_n = 200
    particles_x = np.random.uniform(0, 255, particles_n)
    particles_y = np.random.uniform(0, 1, particles_n)
    particles_z = np.random.uniform(0, 1.8, particles_n) 
    
    fig.add_trace(
        go.Scatter3d(
            x=particles_x,
            y=particles_y,
            z=particles_z,
            mode='markers',
            marker=dict(
                size=1.5,
                color=particles_z,
                colorscale=[
                    [0, '#00ffff'],
                    [0.5, '#ff00ff'],
                    [1, '#ffff00']
                ],
                opacity=0.5
            ),
            hoverinfo='none',
            name='Data Particles'
        )
    )
    
    # Configure layout for cyberpunk aesthetic
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="BYTE VALUE (HEX)",
                    font=dict(family="monospace", size=12, color="#00ffff")
                ),
                range=[0, 255],
                tickvals=[0, 32, 64, 96, 128, 160, 192, 224, 255],
                ticktext=['00h', '20h', '40h', '60h', '80h', 'A0h', 'C0h', 'E0h', 'FFh'],
                tickfont=dict(color="#00ffff"),
                gridcolor='rgba(0,255,255,0.2)',
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.8)',
            ),
            yaxis=dict(
                title=dict(
                    text="DATA DIMENSION",
                    font=dict(family="monospace", size=12, color="#ff00ff")
                ),
                showticklabels=False,
                gridcolor='rgba(255,0,255,0.2)',
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.8)',
            ),
            zaxis=dict(
                title=dict(
                    text="FREQUENCY FACTOR",
                    font=dict(family="monospace", size=12, color="#ffff00")
                ),
                range=[0, 2],
                tickfont=dict(color="#ffff00"),
                gridcolor='rgba(255,255,0,0.2)',
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.8)',
            ),
            # Position the camera based on lower_staging parameter
            camera=dict(
                eye=dict(x=1.8, y=1.2, z=1.5 if not lower_staging else 1.8),
                center=dict(x=0, y=0, z=-0.2 if lower_staging else 0),
                up=dict(x=0, y=0, z=1)
            ),
        ),
        title=dict(
            text="BYTE FREQUENCY ANALYSIS MATRIX",
            font=dict(size=24, color="#00ffff", family="monospace"),
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#00ffff", family="monospace"),
            bgcolor="rgba(0,0,20,0.7)",
            bordercolor="#ff00ff",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    # Add 2D annotation with key statistics at the bottom for better visibility
    fig.add_annotation(
        x=0.5,  # Centered horizontally 
        y=0.02,  # Very bottom
        text=f"<b>MAX BYTE:</b> {int(max_byte)} ({max_byte:02X}h) | <b>PEAK FREQ:</b> {freq_max:.4f} | <b>ANOMALY INDEX:</b> {freq_std/freq_avg:.2f}",
        showarrow=False,
        font=dict(
            family="monospace",
            size=14,
            color="#00ffff"
        ),
        align="center",
        bgcolor="rgba(0,0,30,0.7)",
        bordercolor="#ff00ff",
        borderwidth=2,
        borderpad=6,
        xref="paper",
        yref="paper"
    )
    
    return fig

def create_strings_visualization(strings, max_strings=300):
    """
    Create a cyberpunk word map (word cloud) visualization for extracted strings.
    
    Args:
        strings: List of extracted strings
        max_strings: Maximum number of strings to include
    
    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    import numpy as np
    from math import pi, cos, sin, sqrt
    import random
    
    # Limit number of strings and filter out very short strings for the cloud
    filtered_strings = [s for s in strings if len(s) > 1]
    if len(filtered_strings) > max_strings:
        filtered_strings = filtered_strings[:max_strings]
    
    num_strings = len(filtered_strings)
    
    if num_strings == 0:
        # Create empty visualization with message
        fig = go.Figure()
        fig.add_annotation(
            text="NO STRINGS FOUND",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(family="monospace", size=20, color="#ff00ff")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Count string frequencies for font sizing
    string_counts = {}
    for s in filtered_strings:
        s_upper = s.upper()  # Convert to uppercase for consistency
        if s_upper in string_counts:
            string_counts[s_upper] += 1
        else:
            string_counts[s_upper] = 1
    
    # Get unique strings
    unique_strings = list(string_counts.keys())
    
    # Sort strings by frequency for better placement (more frequent = more central)
    sorted_strings = sorted(unique_strings, key=lambda s: string_counts[s], reverse=True)
    
    # Define color palette for cyberpunk aesthetic
    colors = [
        "#ff00ff",  # Magenta
        "#00ffff",  # Cyan
        "#ffff00",  # Yellow
        "#ff007f",  # Pink
        "#7f00ff",  # Purple
        "#00ff7f",  # Teal
    ]
    
    # Create outer ring for visual boundary
    circle_points = 500
    circle_angles = np.linspace(0, 2*pi, circle_points, endpoint=True)
    radius = 1.0
    x_circle = [radius * cos(angle) for angle in circle_angles]
    y_circle = [radius * sin(angle) for angle in circle_angles]
    
    fig.add_trace(go.Scatter(
        x=x_circle,
        y=y_circle,
        mode="lines",
        line=dict(
            color="#ff00ff",
            width=3
        ),
        hoverinfo="none",
        showlegend=False
    ))
    
    # Add glowing effect to outer ring
    for i in range(3):
        glow_radius = radius * (1 + 0.02 * (i+1))
        x_glow = [glow_radius * cos(angle) for angle in circle_angles]
        y_glow = [glow_radius * sin(angle) for angle in circle_angles]
        
        fig.add_trace(go.Scatter(
            x=x_glow,
            y=y_glow,
            mode="lines",
            line=dict(
                color=f"rgba(0,255,255,{0.3 - i*0.1})" if i % 2 == 0 else f"rgba(255,0,255,{0.3 - i*0.1})",
                width=2
            ),
            hoverinfo="none",
            showlegend=False
        ))
    
    # Create main "STRINGS" title in center
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="text",
        text=["STRINGS"],
        textfont=dict(
            family="monospace",
            size=42,
            color="#ff00ff"
        ),
        hoverinfo="none",
        showlegend=False
    ))
    
    # Add subtitle text
    fig.add_trace(go.Scatter(
        x=[0],
        y=[-0.2],
        mode="text",
        text=["A REFINED TEXTED TEXT"],
        textfont=dict(
            family="monospace",
            size=14,
            color="#ffffff"
        ),
        hoverinfo="none",
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[0],
        y=[-0.35],
        mode="text",
        text=["IN STEGOGRAPHIC ANALYSIS"],
        textfont=dict(
            family="monospace",
            size=12,
            color="#00ffff"
        ),
        hoverinfo="none",
        showlegend=False
    ))
    
    # Create word cloud layout
    # Fix random seed for reproducible layout
    random.seed(42)
    np.random.seed(42)
    
    # Track placed words to avoid overlap
    placed_words = []
    
    # Create word placement with different sizes and colors
    for i, string in enumerate(sorted_strings):
        # Skip common words and very short ones
        if string.lower() in ['the', 'and', 'a', 'of', 'in', 'to'] or len(string) <= 1:
            continue
            
        count = string_counts[string]
        
        # Scale font size based on frequency and length
        # More frequent words and longer words get bigger fonts
        base_size = 8 + min(20, count * 3 + len(string) // 3)
        
        # More frequent words should be closer to center
        freq_factor = min(1.0, count / 10)
        max_radius = 0.9 * (1 - freq_factor * 0.7)
        min_radius = 0.1
        
        # Try multiple positions to find one without overlap
        for attempt in range(50):
            # Get random position within the circle
            if i < 5 and attempt == 0:
                # Place most frequent words in specific positions
                angles = [0, pi/4, pi/2, 3*pi/4, pi]
                r = 0.2 + (i * 0.1)
                theta = angles[i % len(angles)]
            else:
                # Random position for other words
                r = min_radius + (max_radius - min_radius) * random.random()
                theta = 2 * pi * random.random()
            
            x = r * cos(theta)
            y = r * sin(theta)
            
            # Choose color based on position in the circle
            color_idx = int((theta / (2*pi)) * len(colors))
            color = colors[color_idx % len(colors)]
            
            # Check for overlap with already placed words
            overlap = False
            for px, py, ps in placed_words:
                # Calculate distance between word centers
                distance = sqrt((px - x)**2 + (py - y)**2)
                # Minimum distance is based on font sizes
                min_distance = (base_size + ps) / 100
                if distance < min_distance:
                    overlap = True
                    break
            
            if not overlap:
                placed_words.append((x, y, base_size))
                
                # Add word to figure
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode="text",
                    text=[string],
                    textfont=dict(
                        family="monospace",
                        size=base_size,
                        color=color
                    ),
                    textposition="middle center",
                    hoverinfo="text",
                    hovertext=f"{string} (found {count} times)",
                    showlegend=False
                ))
                
                break
    
    # Create background grid effect
    grid_spacing = 0.1
    grid_color = "rgba(0,255,255,0.1)"
    
    # Vertical grid lines
    for x in np.arange(-1.0, 1.1, grid_spacing):
        fig.add_shape(
            type="line",
            x0=x, y0=-1.0,
            x1=x, y1=1.0,
            line=dict(
                color=grid_color,
                width=1
            )
        )
    
    # Horizontal grid lines
    for y in np.arange(-1.0, 1.1, grid_spacing):
        fig.add_shape(
            type="line",
            x0=-1.0, y0=y,
            x1=1.0, y1=y,
            line=dict(
                color=grid_color,
                width=1
            )
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.2, 1.2]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            range=[-1.2, 1.2]
        )
    )
    
    return fig

def format_hex_dump(hex_dump):
    """Format hex dump as simple plain text - no HTML."""
    # Just return the plain hex dump without any HTML formatting
    return hex_dump

def create_detailed_view(plot_figure, title):
    """Create a detailed view layout for a plot with enhanced cyberpunk aesthetics."""
    # Update the figure to be larger and more detailed
    plot_figure.update_layout(
        width=1200,
        height=800,
        title=dict(
            text=title.upper(),
            font=dict(
                size=28, 
                color='#00ffff',
                family='monospace'
            ),
            x=0.5,
            y=0.98
        ),
        showlegend=True,
        legend=dict(
            font=dict(
                color='#00ffff',
                family='monospace',
                size=12
            ),
            bgcolor='rgba(0,0,20,0.8)',
            bordercolor='#ff00ff',
            borderwidth=2,
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=20, r=20, t=80, b=20),
        
        # Add annotations and custom styling elements
        annotations=[
            # Add corner markers to give it a targeting/scanning look
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text='⌜',
                showarrow=False,
                font=dict(size=24, color='#00ffff')
            ),
            dict(
                x=0.98, y=0.98,
                xref='paper', yref='paper',
                text='⌝',
                showarrow=False,
                font=dict(size=24, color='#00ffff')
            ),
            dict(
                x=0.02, y=0.02,
                xref='paper', yref='paper',
                text='⌞',
                showarrow=False,
                font=dict(size=24, color='#00ffff')
            ),
            dict(
                x=0.98, y=0.02,
                xref='paper', yref='paper',
                text='⌟',
                showarrow=False,
                font=dict(size=24, color='#00ffff')
            ),
            
            # Add timestamp and analysis marking
            dict(
                x=0.98, y=0.94,
                xref='paper', yref='paper',
                text=f'T:{datetime.datetime.now().strftime("%H:%M:%S")}',
                showarrow=False,
                font=dict(size=14, color='#ff00ff', family='monospace'),
                align='right'
            ),
            dict(
                x=0.02, y=0.94,
                xref='paper', yref='paper',
                text='DEEP ANAL MATRIX',
                showarrow=False,
                font=dict(size=14, color='#ffff00', family='monospace'),
                align='left'
            ),
            
            # Add scanning line annotation
            dict(
                x=0.5, y=0.94,
                xref='paper', yref='paper',
                text='[ DETAILED ANALYSIS MODE ]',
                showarrow=False,
                font=dict(size=14, color='#00ffff', family='monospace'),
                align='center'
            )
        ],
        
        # Update 3D scene configuration for better visualization
        scene=dict(
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                projection=dict(type='perspective')
            ),
            aspectratio=dict(x=1, y=1, z=0.7)
        )
    )
    
    # Add shapes for cyberpunk border effect
    plot_figure.update_layout(
        shapes=[
            # Horizontal lines (top and bottom)
            dict(
                type="line", xref="paper", yref="paper",
                x0=0.01, y0=0.99, x1=0.99, y1=0.99,
                line=dict(color="#00ffff", width=2)
            ),
            dict(
                type="line", xref="paper", yref="paper",
                x0=0.01, y0=0.01, x1=0.99, y1=0.01,
                line=dict(color="#00ffff", width=2)
            ),
            
            # Vertical lines (left and right)
            dict(
                type="line", xref="paper", yref="paper",
                x0=0.01, y0=0.01, x1=0.01, y1=0.99,
                line=dict(color="#ff00ff", width=2)
            ),
            dict(
                type="line", xref="paper", yref="paper",
                x0=0.99, y0=0.01, x1=0.99, y1=0.99,
                line=dict(color="#ff00ff", width=2)
            ),
        ]
    )
    
    return plot_figure