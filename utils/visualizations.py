import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
import datetime
from plotly.subplots import make_subplots
from PIL import Image
import streamlit as st
from io import BytesIO

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

def create_channel_analysis_visualization(image_path, channel='red'):
    """Create RGB channel analysis visualization showing noise patterns and channel data."""
    try:
        # Open and process image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        pixels = np.array(img)
        
        # Resize if too large for performance
        if pixels.shape[0] > 500 or pixels.shape[1] > 500:
            max_dim = max(pixels.shape[0], pixels.shape[1])
            scale = 500 / max_dim
            new_height = int(pixels.shape[0] * scale)
            new_width = int(pixels.shape[1] * scale)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            pixels = np.array(img_resized)
        
        # Extract channel data
        channel_map = {'red': 0, 'green': 1, 'blue': 2}
        channel_idx = channel_map.get(channel.lower(), 0)
        
        # Get channel data
        channel_data = pixels[:, :, channel_idx]
        
        # Create noise analysis
        noise_pattern = analyze_channel_noise(channel_data)
        
        # Detect anomalies
        anomalies = detect_channel_anomalies(channel_data, noise_pattern)
        
        # Create annotated visualization
        annotated_plot, annotations = create_annotated_channel_plot(
            channel_data, anomalies, channel.capitalize()
        )
        
        # Create visualization data for both original and noise patterns
        return {
            'original': channel_data,
            'noise': noise_pattern,
            'annotated_plot': annotated_plot,
            'annotations': annotations,
            'anomalies': anomalies,
            'channel': channel,
            'stats': calculate_channel_stats(channel_data)
        }
        
    except Exception as e:
        st.error(f"Error processing {channel} channel: {str(e)}")
        return None

def analyze_channel_noise(channel_data):
    """Analyze noise patterns in a single channel and detect anomalies."""
    # Apply various noise detection techniques
    height, width = channel_data.shape
    
    # 1. High-pass filter to detect edges and noise
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    # Apply convolution manually for edge detection
    noise_pattern = np.zeros_like(channel_data, dtype=float)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Apply kernel to detect high-frequency patterns
            region = channel_data[i-1:i+2, j-1:j+2]
            noise_pattern[i, j] = np.sum(region * kernel)
    
    # Normalize and enhance contrast
    noise_pattern = np.abs(noise_pattern)
    if noise_pattern.max() > 0:
        noise_pattern = noise_pattern / noise_pattern.max() * 255
    
    return noise_pattern.astype(np.uint8)

def detect_channel_anomalies(channel_data, noise_pattern):
    """Detect specific anomalies in channel data and return their locations and types."""
    height, width = channel_data.shape
    anomalies = []
    
    # 1. Detect high-contrast regions (potential LSB manipulation)
    high_contrast_threshold = np.percentile(noise_pattern, 95)
    high_contrast_mask = noise_pattern > high_contrast_threshold
    
    # Find connected components of high contrast
    from scipy.ndimage import label
    try:
        labeled_array, num_features = label(high_contrast_mask)
        
        for i in range(1, num_features + 1):
            coords = np.where(labeled_array == i)
            if len(coords[0]) > 20:  # Only significant regions
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
                size = len(coords[0])
                
                anomalies.append({
                    'type': 'high_contrast',
                    'center': (center_x, center_y),
                    'size': size,
                    'severity': min(size / 100, 1.0),
                    'description': 'High-contrast region detected',
                    'recommendations': [
                        'Extract LSB data from this region',
                        'Analyze bit patterns in surrounding pixels',
                        'Check for hidden message boundaries'
                    ]
                })
    except ImportError:
        # Fallback without scipy
        pass
    
    # 2. Detect regular patterns (potential algorithmic hiding)
    # Look for repeating patterns in small windows
    window_size = 8
    pattern_threshold = 0.8
    
    for i in range(0, height - window_size, window_size // 2):
        for j in range(0, width - window_size, window_size // 2):
            window = channel_data[i:i+window_size, j:j+window_size]
            
            # Check for repetitive patterns
            variance = np.var(window)
            mean_val = np.mean(window)
            
            # Look for suspicious uniformity or regular patterns
            if variance < 5 and mean_val > 50:  # Very uniform region
                anomalies.append({
                    'type': 'uniform_region',
                    'center': (j + window_size//2, i + window_size//2),
                    'size': window_size * window_size,
                    'severity': 0.6,
                    'description': 'Suspiciously uniform region',
                    'recommendations': [
                        'Check for steganographic algorithms',
                        'Analyze adjacent regions for similar patterns',
                        'Test with different extraction methods'
                    ]
                })
    
    # 3. Detect entropy hotspots
    entropy_window = 16
    for i in range(0, height - entropy_window, entropy_window // 2):
        for j in range(0, width - entropy_window, entropy_window // 2):
            window = channel_data[i:i+entropy_window, j:j+entropy_window]
            local_entropy = calculate_local_entropy(window)
            
            if local_entropy > 7.5:  # High local entropy
                anomalies.append({
                    'type': 'entropy_hotspot',
                    'center': (j + entropy_window//2, i + entropy_window//2),
                    'size': entropy_window * entropy_window,
                    'severity': min((local_entropy - 7.0) / 1.0, 1.0),
                    'description': f'High entropy region ({local_entropy:.2f})',
                    'recommendations': [
                        'High entropy suggests encrypted or compressed data',
                        'Try frequency analysis on this region',
                        'Check for cryptographic signatures'
                    ]
                })
    
    # Sort by severity (most severe first)
    anomalies.sort(key=lambda x: x['severity'], reverse=True)
    
    return anomalies[:10]  # Return top 10 anomalies

def calculate_local_entropy(data):
    """Calculate entropy for a small data window."""
    hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 256))
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0
    return -np.sum(hist * np.log2(hist))

def create_annotated_channel_plot(channel_data, anomalies, channel_name):
    """Create an annotated Plotly visualization showing detected anomalies with circles and recommendations."""
    height, width = channel_data.shape
    
    # Create the base heatmap
    fig = go.Figure()
    
    # Add the channel data as a heatmap
    fig.add_trace(go.Heatmap(
        z=channel_data,
        colorscale='gray',
        showscale=False,
        hoverinfo='none'
    ))
    
    # Define colors for different anomaly types
    colors = {
        'high_contrast': '#ff0000',
        'uniform_region': '#ffff00',
        'entropy_hotspot': '#00ffff'
    }
    
    annotations = []
    
    # Add circles and annotations for each anomaly
    for i, anomaly in enumerate(anomalies):
        x, y = anomaly['center']
        anomaly_type = anomaly['type']
        severity = anomaly['severity']
        
        # Circle size based on severity and anomaly size
        radius = max(10, min(40, np.sqrt(anomaly['size']) * severity * 1.5))
        
        # Create circle points
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = x + radius * np.cos(theta)
        circle_y = y + radius * np.sin(theta)
        
        # Add circle outline
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(
                color=colors.get(anomaly_type, '#ffffff'),
                width=3
            ),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add numbered annotation
        fig.add_trace(go.Scatter(
            x=[x + radius + 15],
            y=[y - radius - 15],
            mode='markers+text',
            marker=dict(
                size=20,
                color='black',
                line=dict(
                    color=colors.get(anomaly_type, '#ffffff'),
                    width=2
                )
            ),
            text=[str(i + 1)],
            textfont=dict(
                color=colors.get(anomaly_type, '#ffffff'),
                size=12
            ),
            showlegend=False,
            hovertemplate=f'<b>Anomaly {i+1}</b><br>' +
                         f'Type: {anomaly_type.replace("_", " ").title()}<br>' +
                         f'Severity: {severity:.1f}<br>' +
                         f'{anomaly["description"]}<extra></extra>'
        ))
        
        # Add arrow from annotation to anomaly center
        fig.add_annotation(
            x=x, y=y,
            ax=x + radius + 15, ay=y - radius - 15,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=colors.get(anomaly_type, '#ffffff'),
            showarrow=True
        )
        
        # Store annotation info for legend
        annotations.append({
            'number': i + 1,
            'type': anomaly_type.replace('_', ' ').title(),
            'severity': f"{severity:.1f}",
            'description': anomaly['description'],
            'recommendations': anomaly['recommendations']
        })
    
    # Update layout for cyberpunk style
    fig.update_layout(
        title={
            'text': f'{channel_name} Channel - Anomaly Detection',
            'font': {'color': '#00ffff', 'size': 18}
        },
        paper_bgcolor='black',
        plot_bgcolor='black',
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-10, width + 10]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor='x',
            scaleratio=1,
            autorange='reversed',  # Flip Y axis to match image coordinates
            range=[-10, height + 10]
        ),
        width=800,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig, annotations

def calculate_channel_stats(channel_data):
    """Calculate statistics for a channel."""
    return {
        'mean': float(np.mean(channel_data)),
        'std': float(np.std(channel_data)),
        'entropy': calculate_channel_entropy(channel_data),
        'histogram_peaks': count_histogram_peaks(channel_data)
    }

def calculate_channel_entropy(channel_data):
    """Calculate entropy of channel data."""
    hist, _ = np.histogram(channel_data.flatten(), bins=256, range=(0, 256))
    hist = hist / np.sum(hist)  # Normalize
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)

def count_histogram_peaks(channel_data):
    """Count peaks in channel histogram."""
    hist, _ = np.histogram(channel_data.flatten(), bins=256, range=(0, 256))
    peaks = 0
    for i in range(1, 255):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks += 1
    return peaks

def create_channel_comparison_plot(red_stats, green_stats, blue_stats):
    """Create comparison plot of all three channels."""
    channels = ['Red', 'Green', 'Blue']
    entropies = [red_stats['entropy'], green_stats['entropy'], blue_stats['entropy']]
    means = [red_stats['mean'], green_stats['mean'], blue_stats['mean']]
    stds = [red_stats['std'], green_stats['std'], blue_stats['std']]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Channel Entropy', 'Mean Values', 'Standard Deviation'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Add entropy plot
    fig.add_trace(go.Bar(
        x=channels,
        y=entropies,
        marker_color=['#ff0000', '#00ff00', '#0000ff'],
        name='Entropy'
    ), row=1, col=1)
    
    # Add means plot
    fig.add_trace(go.Bar(
        x=channels,
        y=means,
        marker_color=['#ff0000', '#00ff00', '#0000ff'],
        name='Mean'
    ), row=1, col=2)
    
    # Add standard deviation plot
    fig.add_trace(go.Bar(
        x=channels,
        y=stds,
        marker_color=['#ff0000', '#00ff00', '#0000ff'],
        name='Std Dev'
    ), row=1, col=3)
    
    fig.update_layout(
        title={
            'text': 'RGB CHANNEL ANALYSIS COMPARISON',
            'font': {'color': '#00ffff', 'size': 20}
        },
        showlegend=False,
        **create_cyberpunk_theme()
    )
    
    return fig


# ============================================================================
# NEW VISUALIZATION MODULES [UPDATED]
# ============================================================================

def create_byte_frequency_plot_upgraded(image_path, mode='3d'):
    """
    [UPDATED] Upgraded Byte Frequency Module with 2D/3D toggle.
    
    Args:
        image_path: Path to image file
        mode: '2d' for heatmap, '3d' for bar graph (default: '3d')
    
    Returns:
        Plotly figure object
    """
    # Read image and extract byte values
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Flatten to get all byte values
    if len(img_array.shape) == 3:
        bytes_data = img_array.reshape(-1)
    else:
        bytes_data = img_array.flatten()
    
    # Calculate frequency distribution
    byte_counts = np.bincount(bytes_data, minlength=256)
    byte_values = np.arange(256)
    
    if mode == '2d':
        # 2D Heatmap mode
        # Reshape into 16x16 grid for visualization
        heatmap_data = byte_counts.reshape(16, 16)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            colorscale=[
                [0, '#000014'],
                [0.2, '#00ffff'],
                [0.4, '#0080ff'],
                [0.6, '#ff00ff'],
                [0.8, '#ff0080'],
                [1, '#ffff00']
            ],
            colorbar=dict(
                title=dict(
                    text="FREQUENCY",
                    font=dict(color='#00ffff', family='monospace')
                ),
                tickfont=dict(color='#00ffff'),
                len=0.7
            ),
            hovertemplate='Byte: %{x},%{y}<br>Freq: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="BYTE FREQUENCY HEATMAP [2D MODE]",
                font=dict(size=24, color='#00ffff', family='monospace'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(text="BYTE COLUMN", font=dict(color='#00ffff')),
                gridcolor='rgba(0,255,255,0.2)',
                tickfont=dict(color='#00ffff')
            ),
            yaxis=dict(
                title=dict(text="BYTE ROW", font=dict(color='#ff00ff')),
                gridcolor='rgba(255,0,255,0.2)',
                tickfont=dict(color='#ff00ff')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,20,0.9)',
            font=dict(family='monospace'),
            height=600
        )
        
    else:
        # 3D Bar Graph mode (using Scatter3d with marker sizes)
        fig = go.Figure()
        
        # Normalize frequencies for better visualization
        freq_norm = byte_counts / np.max(byte_counts) if np.max(byte_counts) > 0 else byte_counts
        
        # Create 3D bars using Scatter3d markers
        fig.add_trace(go.Scatter3d(
            x=byte_values,
            y=np.zeros_like(byte_values),
            z=freq_norm,
            mode='markers',
            marker=dict(
                size=8,
                color=byte_values,
                colorscale=[
                    [0, '#00ffff'],
                    [0.3, '#0080ff'],
                    [0.5, '#ff00ff'],
                    [0.7, '#ff0080'],
                    [1, '#ffff00']
                ],
                colorbar=dict(
                    title=dict(text="BYTE VALUE", font=dict(color='#00ffff')),
                    tickfont=dict(color='#00ffff')
                ),
                symbol='square',
                line=dict(color='#ffffff', width=0.5)
            ),
            hovertemplate='Byte: %{x:02X}h<br>Frequency: %{z:.3f}<extra></extra>',
            name='Byte Frequency'
        ))
        
        # Add vertical lines to create bar effect
        for i in range(0, 256, 4):  # Sample every 4th byte to avoid clutter
            if freq_norm[i] > 0.05:  # Only show significant bars
                fig.add_trace(go.Scatter3d(
                    x=[i, i],
                    y=[0, 0],
                    z=[0, freq_norm[i]],
                    mode='lines',
                    line=dict(
                        color=byte_values[i],
                        width=3,
                        colorscale=[
                            [0, '#00ffff'],
                            [0.5, '#ff00ff'],
                            [1, '#ffff00']
                        ]
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title=dict(
                text="BYTE FREQUENCY ANALYSIS [3D MODE]",
                font=dict(size=24, color='#00ffff', family='monospace'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="BYTE VALUE", font=dict(color='#00ffff')),
                    gridcolor='rgba(0,255,255,0.2)',
                    tickfont=dict(color='#00ffff'),
                    showbackground=True,
                    backgroundcolor='rgba(0,0,20,0.9)'
                ),
                yaxis=dict(
                    title=dict(text="", font=dict(color='#ff00ff')),
                    showticklabels=False,
                    gridcolor='rgba(255,0,255,0.2)',
                    showbackground=True,
                    backgroundcolor='rgba(0,0,20,0.9)'
                ),
                zaxis=dict(
                    title=dict(text="FREQUENCY", font=dict(color='#ffff00')),
                    gridcolor='rgba(255,255,0,0.2)',
                    tickfont=dict(color='#ffff00'),
                    showbackground=True,
                    backgroundcolor='rgba(0,0,20,0.9)'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700
        )
    
    return fig


def create_bitplane_visualizer(image_path, group_mode='all'):
    """
    Bitplane Visualizer - Extract and display all 24 bitplanes (8 per RGB channel).
    
    Args:
        image_path: Path to image file
        group_mode: 'all' (all 24), 'lsb' (bits 0-3), 'msb' (bits 4-7)
    
    Returns:
        Plotly figure object with bitplane visualizations
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Determine which bits to display
    if group_mode == 'lsb':
        bit_range = range(4)  # Bits 0-3
        title_suffix = "[LSB LAYERS 0-3]"
    elif group_mode == 'msb':
        bit_range = range(4, 8)  # Bits 4-7
        title_suffix = "[MSB LAYERS 4-7]"
    else:
        bit_range = range(8)  # All bits 0-7
        title_suffix = "[ALL 24 BITPLANES]"
    
    # Extract bitplanes
    channels = ['R', 'G', 'B']
    channel_colors = ['#ff0000', '#00ff00', '#0000ff']
    
    # Create subplot grid
    rows = len(bit_range)
    cols = 3
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'{ch} Bit {b}' for b in bit_range for ch in channels],
        vertical_spacing=0.02,
        horizontal_spacing=0.01,
        specs=[[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Extract and plot each bitplane
    for bit_idx, bit in enumerate(bit_range):
        for ch_idx, channel_name in enumerate(channels):
            # Extract bitplane
            channel_data = img_array[:, :, ch_idx]
            bitplane = (channel_data >> bit) & 1
            
            # Add to subplot
            row = bit_idx + 1
            col = ch_idx + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=bitplane,
                    colorscale=[[0, '#000000'], [1, channel_colors[ch_idx]]],
                    showscale=False,
                    hovertemplate=f'{channel_name} Bit {bit}<br>Value: %{{z}}<extra></extra>'
                ),
                row=row, col=col
            )
    
    # Update all axes to remove ticks
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    
    fig.update_layout(
        title=dict(
            text=f"BITPLANE ANALYSIS {title_suffix}",
            font=dict(size=20, color='#00ffff', family='monospace'),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,20,0.9)',
        height=200 * rows,
        showlegend=False,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    
    return fig


def create_rgb_3d_scatter(image_path, sample_size=5000, enable_density=True):
    """
    RGB 3D Scatter Plot - Map each pixel's RGB values into 3D color space.
    
    Args:
        image_path: Path to image file
        sample_size: Number of pixels to sample (for performance)
        enable_density: Apply density smoothing and color fade
    
    Returns:
        Plotly figure object
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Flatten to get all pixels
    pixels = img_array.reshape(-1, 3)
    
    # Sample if too many pixels
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]
    
    # Extract RGB coordinates
    r_vals = pixels[:, 0]
    g_vals = pixels[:, 1]
    b_vals = pixels[:, 2]
    
    # Create color array for markers (actual pixel colors)
    pixel_colors = [f'rgb({r},{g},{b})' for r, g, b in pixels]
    
    # Calculate density if enabled
    if enable_density:
        # Simple density estimation: count nearby points
        from scipy.spatial import cKDTree
        tree = cKDTree(pixels)
        density = np.array([len(tree.query_ball_point(p, r=30)) for p in pixels])
        # Convert density to color intensity instead of opacity
        marker_opacity = 0.6  # Fixed opacity
        marker_size = 2 + (density / np.max(density)) * 4
    else:
        marker_opacity = 0.6
        marker_size = 3
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=r_vals,
        y=g_vals,
        z=b_vals,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=pixel_colors,
            opacity=marker_opacity,
            line=dict(width=0)
        ),
        hovertemplate='R: %{x}<br>G: %{y}<br>B: %{z}<extra></extra>',
        name='Pixels'
    ))
    
    # Add axis reference lines
    max_val = 255
    axis_line_color = 'rgba(255,255,255,0.3)'
    
    # R axis
    fig.add_trace(go.Scatter3d(
        x=[0, max_val], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='#ff0000', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # G axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, max_val], z=[0, 0],
        mode='lines',
        line=dict(color='#00ff00', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # B axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, max_val],
        mode='lines',
        line=dict(color='#0000ff', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text="RGB COLOR SPACE DISTRIBUTION",
            font=dict(size=24, color='#00ffff', family='monospace'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="RED CHANNEL", font=dict(color='#ff0000')),
                range=[0, 255],
                gridcolor='rgba(255,0,0,0.2)',
                tickfont=dict(color='#ff0000'),
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.9)'
            ),
            yaxis=dict(
                title=dict(text="GREEN CHANNEL", font=dict(color='#00ff00')),
                range=[0, 255],
                gridcolor='rgba(0,255,0,0.2)',
                tickfont=dict(color='#00ff00'),
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.9)'
            ),
            zaxis=dict(
                title=dict(text="BLUE CHANNEL", font=dict(color='#0000ff')),
                range=[0, 255],
                gridcolor='rgba(0,0,255,0.2)',
                tickfont=dict(color='#0000ff'),
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.9)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700
    )
    
    return fig


def create_entropy_terrain_map(image_path, block_size=16):
    """
    Entropy Terrain Map - Block-based Shannon entropy visualization.
    
    Args:
        image_path: Path to image file
        block_size: Size of blocks for entropy calculation (default: 16x16)
    
    Returns:
        Plotly figure object
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale for entropy
    img_array = np.array(img)
    height, width = img_array.shape
    
    # Calculate number of blocks
    n_blocks_h = height // block_size
    n_blocks_w = width // block_size
    
    # Initialize entropy map
    entropy_map = np.zeros((n_blocks_h, n_blocks_w))
    
    # Calculate Shannon entropy for each block
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            # Extract block
            block = img_array[i*block_size:(i+1)*block_size, 
                             j*block_size:(j+1)*block_size]
            
            # Calculate histogram
            hist, _ = np.histogram(block.flatten(), bins=256, range=(0, 256))
            hist = hist / np.sum(hist)  # Normalize
            hist = hist[hist > 0]  # Remove zeros
            
            # Shannon entropy
            if len(hist) > 0:
                entropy_map[i, j] = -np.sum(hist * np.log2(hist))
    
    # Create 3D surface plot
    x = np.arange(n_blocks_w) * block_size
    y = np.arange(n_blocks_h) * block_size
    X, Y = np.meshgrid(x, y)
    
    fig = go.Figure()
    
    # Main terrain surface
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=entropy_map,
        colorscale=[
            [0, '#000014'],
            [0.2, '#0000ff'],
            [0.4, '#00ffff'],
            [0.6, '#00ff00'],
            [0.8, '#ffff00'],
            [1, '#ff0000']
        ],
        colorbar=dict(
            title=dict(
                text="ENTROPY",
                font=dict(color='#00ffff', family='monospace')
            ),
            tickfont=dict(color='#00ffff'),
            len=0.7
        ),
        hovertemplate='Block: (%{x}, %{y})<br>Entropy: %{z:.3f}<extra></extra>',
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.9,
            roughness=0.3,
            fresnel=0.5
        )
    ))
    
    # Add contour lines at base
    fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=entropy_map,
        colorscale=[[0, 'rgba(0,255,255,0.3)'], [1, 'rgba(255,0,255,0.3)']],
        showscale=False,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=8, color='#00ffff')
        ),
        hoverinfo='skip'
    ))
    
    # Calculate statistics
    avg_entropy = np.mean(entropy_map)
    max_entropy = np.max(entropy_map)
    min_entropy = np.min(entropy_map)
    std_entropy = np.std(entropy_map)
    
    fig.update_layout(
        title=dict(
            text=f"ENTROPY TERRAIN MAP (Block Size: {block_size}x{block_size})",
            font=dict(size=20, color='#00ffff', family='monospace'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="X COORDINATE", font=dict(color='#00ffff')),
                gridcolor='rgba(0,255,255,0.2)',
                tickfont=dict(color='#00ffff'),
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.9)'
            ),
            yaxis=dict(
                title=dict(text="Y COORDINATE", font=dict(color='#ff00ff')),
                gridcolor='rgba(255,0,255,0.2)',
                tickfont=dict(color='#ff00ff'),
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.9)'
            ),
            zaxis=dict(
                title=dict(text="SHANNON ENTROPY", font=dict(color='#ffff00')),
                gridcolor='rgba(255,255,0,0.2)',
                tickfont=dict(color='#ffff00'),
                showbackground=True,
                backgroundcolor='rgba(0,0,20,0.9)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        annotations=[
            dict(
                text=f"AVG: {avg_entropy:.3f} | MAX: {max_entropy:.3f} | MIN: {min_entropy:.3f} | STD: {std_entropy:.3f}",
                x=0.5,
                y=0.02,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(color='#00ffff', size=12, family='monospace'),
                bgcolor='rgba(0,0,30,0.8)',
                bordercolor='#00ffff',
                borderwidth=2,
                borderpad=8
            )
        ]
    )
    
    return fig


def create_segment_structure_mapper(image_path):
    """
    Segment Structure Mapper - Parse and visualize file format structure.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Plotly figure object showing file structure
    """
    import os
    
    # Read file as binary
    with open(image_path, 'rb') as f:
        file_data = f.read()
    
    file_size = len(file_data)
    
    # Detect file format from magic bytes
    magic_bytes = file_data[:16]
    
    segments = []
    
    # PNG parser
    if file_data[:8] == b'\x89PNG\r\n\x1a\n':
        offset = 8
        while offset < file_size:
            if offset + 8 > file_size:
                break
            
            # Read chunk length and type
            chunk_len = int.from_bytes(file_data[offset:offset+4], 'big')
            chunk_type = file_data[offset+4:offset+8].decode('ascii', errors='replace')
            
            # Total chunk size includes length (4) + type (4) + data + CRC (4)
            total_size = chunk_len + 12
            
            segments.append({
                'type': f'PNG:{chunk_type}',
                'offset': offset,
                'size': total_size,
                'color': '#00ffff' if chunk_type == 'IDAT' else '#ff00ff'
            })
            
            offset += total_size
            
            if chunk_type == 'IEND':
                break
    
    # JPEG parser
    elif file_data[:2] == b'\xff\xd8':
        offset = 0
        while offset < file_size - 1:
            if file_data[offset] == 0xff:
                marker = file_data[offset+1]
                
                # Marker names
                marker_names = {
                    0xd8: 'SOI', 0xd9: 'EOI', 0xda: 'SOS',
                    0xdb: 'DQT', 0xc0: 'SOF0', 0xc4: 'DHT',
                    0xe0: 'APP0', 0xe1: 'APP1', 0xfe: 'COM'
                }
                
                marker_name = marker_names.get(marker, f'{marker:02X}')
                
                # Special markers without length
                if marker in [0xd8, 0xd9]:
                    size = 2
                else:
                    if offset + 3 < file_size:
                        size = int.from_bytes(file_data[offset+2:offset+4], 'big') + 2
                    else:
                        break
                
                segments.append({
                    'type': f'JPEG:{marker_name}',
                    'offset': offset,
                    'size': size,
                    'color': '#ffff00' if marker == 0xda else '#00ff00'
                })
                
                offset += size
            else:
                offset += 1
    
    # Generic format or unknown
    else:
        # Create blocks of the file
        block_size = max(1024, file_size // 20)
        for i in range(0, file_size, block_size):
            segments.append({
                'type': f'Block {i//block_size}',
                'offset': i,
                'size': min(block_size, file_size - i),
                'color': '#ff00ff'
            })
    
    # Create timeline visualization
    fig = go.Figure()
    
    # Add segments as bars
    for seg in segments:
        fig.add_trace(go.Bar(
            x=[seg['size']],
            y=[seg['type']],
            orientation='h',
            marker=dict(
                color=seg['color'],
                line=dict(color='#ffffff', width=1)
            ),
            hovertemplate=f"<b>{seg['type']}</b><br>" +
                         f"Offset: {seg['offset']:,} bytes<br>" +
                         f"Size: {seg['size']:,} bytes<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text=f"FILE STRUCTURE MAP ({os.path.basename(image_path)})",
            font=dict(size=20, color='#00ffff', family='monospace'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="SIZE (bytes)", font=dict(color='#00ffff')),
            gridcolor='rgba(0,255,255,0.2)',
            tickfont=dict(color='#00ffff')
        ),
        yaxis=dict(
            title=dict(text="SEGMENT", font=dict(color='#ff00ff')),
            tickfont=dict(color='#ff00ff', size=10)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,20,0.9)',
        height=max(400, len(segments) * 30),
        barmode='stack',
        annotations=[
            dict(
                text=f"Total Size: {file_size:,} bytes | Segments: {len(segments)}",
                x=0.5,
                y=1.05,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(color='#00ffff', size=14, family='monospace'),
                bgcolor='rgba(0,0,30,0.8)',
                bordercolor='#00ffff',
                borderwidth=2,
                borderpad=6
            )
        ]
    )
    
    return fig