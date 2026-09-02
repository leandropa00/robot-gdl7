import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

"""
Simulador Cinemático de Robot Híbrido SCARA / Articulado de 7 GDL + Pinza (8 DOF totales)

Cadena Cinemática (7 GDL):
1. Rotación de la base (Yaw)                    -> q1 (Rotación en Z0)
2. Inclinación de la articulación inferior (Pitch)-> q2 (Rotación en Y1)
3. Inclinación del codo (Pitch)                 -> q3 (Rotación en Y2)
4. Rotación del cuerpo/antebrazo (Roll)          -> q4 (Rotación en Z3 axial)
5. Inclinación de la cabeza previa a la barra   -> q5 (Rotación en Y4 transversal)
6. Desplazamiento lineal vertical (Prismático)  -> d6 (Translación en Z5 axial, carrera 300 mm)
7. Rotación de la muñeca (Roll)                 -> q7 (Rotación en Z6 axial)

Efector Final (Mecanismo de agarre): 1 GDL
8. Cierre/apertura de la pinza                  -> q_grip (Apertura simétrica de dedos)
"""

# ==============================================================================
# 1. MATRICES DE TRANSFORMACIÓN HOMOGÉNEA Y DENAVIT-HARTENBERG
# ==============================================================================

def dh_matrix(alpha, a, theta, d):
    """
    Matriz de transformación homogénea de Denavit-Hartenberg Modificada (Craig).
    T = Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct,               -st,              0,              a],
        [st * ca,          ct * ca,         -sa,            -d * sa],
        [st * sa,          ct * sa,          ca,             d * ca],
        [0,                0,                0,              1]
    ])

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def trans(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

# ==============================================================================
# 2. MODELO CINEMÁTICO DIRECTO (7 GDL + PINZA)
# ==============================================================================

def get_robot_skeleton_7gdl(q1_deg, q2_deg, q3_deg, q4_deg, q5_deg, d6_cm, q7_deg, q_grip_cm):
    """
    Calcula la cinemática directa de la cadena cinemática de 7 GDL y el mecanismo de la pinza.
    
    Retorna un diccionario con las coordenadas 3D de todos los eslabones, articulaciones,
    barra vertical, motor superior, muñeca y dedos de la pinza.
    """
    q1 = np.radians(q1_deg)
    q2 = np.radians(q2_deg)
    q3 = np.radians(q3_deg)
    q4 = np.radians(q4_deg)
    q5 = np.radians(q5_deg)
    d6 = d6_cm
    q7 = np.radians(q7_deg)
    grip = q_grip_cm

    # Dimensiones físicas de los eslabones (en cm)
    h_pedestal = 14.0      # Base fija metálica
    h_turret   = 8.0       # Altura torreta / hombro
    L_arm1     = 38.0      # Brazo inferior naranja (J1)
    L_arm2     = 30.0      # Antebrazo gris (J2)
    L_housing  = 14.0      # Bloque guía vertical (Z-AXIS 300mm)
    L_rod_total= 42.0      # Longitud total de la barra cromada
    L_wrist    = 4.5       # Muñeca WRIST 4
    L_grip_base= 3.5       # Base naranja de la pinza
    L_finger   = 7.0       # Longitud de las mordazas de agarre

    # Transformación base en suelo
    T_world = np.eye(4)
    T_base_top = trans(0, 0, h_pedestal)

    # 1. Base Yaw (q1) - Rotación alrededor del eje vertical Z
    T1 = T_base_top @ rot_z(q1) @ trans(0, 0, h_turret)

    # 2. Inclinación Hombro (Pitch q2) - Rotación horizontal en Y
    T2 = T1 @ rot_y(q2)
    T_elbow = T2 @ trans(0, 0, L_arm1)

    # 3. Inclinación Codo (Pitch q3) - Rotación horizontal en Y
    T3 = T_elbow @ rot_y(q3)

    # 4. Rotación Antebrazo (Roll q4) - Rotación axial a lo largo del antebrazo (Z local)
    T4 = T3 @ rot_z(q4)
    T_head = T4 @ trans(0, 0, L_arm2)

    # 5. Inclinación Cabeza previa a la barra (Pitch q5) - Rotación transversal en Y
    T5 = T_head @ rot_y(q5)

    # 6. Desplazamiento Lineal Vertical (Prismático / Eje Z d6)
    # Bloque guía / carcasa fijo en T5
    T_guide_top = T5 @ trans(0, 0, L_housing / 2.0)
    T_guide_bot = T5 @ trans(0, 0, -L_housing / 2.0)

    # La barra cromada se desplaza axialmente según d6
    T_rod_center = T5 @ trans(0, 0, -d6)
    T_rod_top    = T_rod_center @ trans(0, 0, L_rod_total / 2.0)
    T_rod_bot    = T_rod_center @ trans(0, 0, -L_rod_total / 2.0)
    T_motor      = T_rod_top @ trans(0, 0, 5.0)

    # 7. Rotación de la muñeca (Roll q7) - Rotación en el extremo inferior de la barra (Z)
    T7 = T_rod_bot @ rot_z(q7)
    T_wrist_flange = T7 @ trans(0, 0, -L_wrist)

    # 8. Efector final: Base de la pinza
    T_palm = T_wrist_flange @ trans(0, 0, -L_grip_base)

    # Dedos de la pinza (apertura y cierre de mordazas)
    half_w = np.clip(grip, 0.4, 10.0) / 2.0
    f1_base = T_palm @ trans(-half_w, 0, 0)
    f1_mid  = f1_base @ trans(0, 0, -L_finger * 0.6)
    f1_tip  = f1_mid @ trans(half_w * 0.25, 0, -L_finger * 0.4)

    f2_base = T_palm @ trans(half_w, 0, 0)
    f2_mid  = f2_base @ trans(0, 0, -L_finger * 0.6)
    f2_tip  = f2_mid @ trans(-half_w * 0.25, 0, -L_finger * 0.4)

    # Pieza sujeta (cilindro entre dedos)
    obj_p1 = T_palm @ trans(-half_w - 1.8, 0, -L_finger * 0.9)
    obj_p2 = T_palm @ trans(half_w + 1.8, 0, -L_finger * 0.9)

    # Tríada de orientación del efector final
    axis_len = 6.0
    ee_pos = T_palm[:3, 3]
    ee_x = ee_pos + T_palm[:3, 0] * axis_len
    ee_y = ee_pos + T_palm[:3, 1] * axis_len
    ee_z = ee_pos + T_palm[:3, 2] * axis_len

    # Cálculo de ángulos de Euler (RPY) del efector final
    R = T_palm[:3, :3]
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll_ee  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch_ee = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw_ee   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll_ee  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch_ee = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw_ee   = 0.0

    return {
        'p_origin': T_world[:3, 3],
        'p_base': T_base_top[:3, 3],
        'p_shoulder': T1[:3, 3],
        'p_elbow': T_elbow[:3, 3],
        'p_head': T_head[:3, 3],
        'p_guide_top': T_guide_top[:3, 3],
        'p_guide_bot': T_guide_bot[:3, 3],
        'p_rod_top': T_rod_top[:3, 3],
        'p_rod_bot': T_rod_bot[:3, 3],
        'p_motor': T_motor[:3, 3],
        'p_wrist': T_wrist_flange[:3, 3],
        'p_palm': T_palm[:3, 3],
        'f1_nodes': np.array([T_palm[:3, 3], f1_base[:3, 3], f1_mid[:3, 3], f1_tip[:3, 3]]),
        'f2_nodes': np.array([T_palm[:3, 3], f2_base[:3, 3], f2_mid[:3, 3], f2_tip[:3, 3]]),
        'obj_nodes': np.array([obj_p1[:3, 3], obj_p2[:3, 3]]),
        'ee_pos': ee_pos,
        'ee_x': ee_x,
        'ee_y': ee_y,
        'ee_z': ee_z,
        'ee_rpy': (roll_ee, pitch_ee, yaw_ee)
    }

# ==============================================================================
# 3. CONFIGURACIÓN VISUAL DEL ENTORNO MATPLOTLIB 3D
# ==============================================================================

fig = plt.figure(figsize=(13, 10.5), facecolor='#1E1E1E')
plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.28)
fig.suptitle('Simulador Robot SCARA/Articulado 7-GDL + Pinza (8 DOF)', 
             color='#FFFFFF', fontsize=13.5, fontweight='bold', y=0.97)

ax = fig.add_subplot(111, projection='3d', facecolor='#181818')

ax.set_xlabel('X (cm)', color='#B0BEC5', labelpad=6)
ax.set_ylabel('Y (cm)', color='#B0BEC5', labelpad=6)
ax.set_zlabel('Z (cm)', color='#B0BEC5', labelpad=6)

ax.set_xlim([-55, 75])
ax.set_ylim([-65, 65])
ax.set_zlim([0, 95])

ax.tick_params(colors='#B0BEC5')
ax.grid(True, linestyle=':', color='#37474F', alpha=0.6)
ax.view_init(elev=22, azim=-55)

# Marcas de suelo industrial (diana de posicionamiento)
t_box = np.array([-30, 30, 30, -30, -30]) + 35
y_box = np.array([-30, -30, 30, 30, -30])
ax.plot(t_box, y_box, np.zeros(5), color='#CFD8DC', linestyle='--', linewidth=1.2, alpha=0.5)
ax.plot([35, 35], [-12, 12], [0, 0], color='#FFFFFF', linewidth=1.2, alpha=0.5)
ax.plot([23, 47], [0, 0], [0, 0], color='#FFFFFF', linewidth=1.2, alpha=0.5)

# Elementos gráficos del robot (inicialización)
line_pedestal, = ax.plot([], [], [], color='#37474F', linewidth=14, solid_capstyle='butt', label='1. Base (Yaw)')
line_turret,   = ax.plot([], [], [], color='#78909C', linewidth=12, solid_capstyle='butt')
line_arm1,     = ax.plot([], [], [], color='#FF6F00', linewidth=10, solid_capstyle='round', label='2. Hombro J1 (Pitch)')
line_arm2,     = ax.plot([], [], [], color='#90A4AE', linewidth=8, solid_capstyle='round', label='3. Codo / 4. Antebrazo (Roll)')
line_guide,    = ax.plot([], [], [], color='#546E7A', linewidth=13, solid_capstyle='butt', label='5. Guía Z (Pitch)')
line_rod,      = ax.plot([], [], [], color='#ECEFF1', linewidth=4.5, solid_capstyle='round', label='6. Barra Prismática Z')
line_motor,    = ax.plot([], [], [], color='#212121', linewidth=11, solid_capstyle='butt')
line_wrist,    = ax.plot([], [], [], color='#607D8B', linewidth=7.5, label='7. Muñeca (Roll)')
line_f1,       = ax.plot([], [], [], color='#FF6F00', linewidth=4.5, label='8. Pinza (Efector)')
line_f2,       = ax.plot([], [], [], color='#FF6F00', linewidth=4.5)

# Ejes de orientación en el efector final (X: Rojo, Y: Verde, Z: Azul)
line_ee_x, = ax.plot([], [], [], color='#F44336', linewidth=2.5, label='Ejes TCP (RGB)')
line_ee_y, = ax.plot([], [], [], color='#4CAF50', linewidth=2.5)
line_ee_z, = ax.plot([], [], [], color='#2196F3', linewidth=2.5)

# Marcadores de centros de articulación
scat_joints = ax.scatter([], [], [], color='#263238', s=120, edgecolors='#ECEFF1', linewidths=1.5, zorder=10)

# Cuadro de información en tiempo real (HUD) en la esquina superior izquierda
info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
                      color='#ECEFF1', fontsize=8.0, family='monospace',
                      bbox=dict(boxstyle='round,pad=0.45', facecolor='#212121', edgecolor='#455A64', alpha=0.9))

# Leyenda en la esquina inferior izquierda (debajo del HUD, sin solaparse nunca)
legend = ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), ncol=1, 
                   fontsize=7.5, facecolor='#212121', edgecolor='#455A64', framealpha=0.9)
for txt in legend.get_texts():
    txt.set_color('#ECEFF1')

# ==============================================================================
# 4. FUNCIÓN DE ACTUALIZACIÓN DEL SIMULADOR
# ==============================================================================

def update(val=0):
    q1 = s_q1.val
    q2 = s_q2.val
    q3 = s_q3.val
    q4 = s_q4.val
    q5 = s_q5.val
    d6 = s_d6.val
    q7 = s_q7.val
    grip = s_grip.val

    geom = get_robot_skeleton_7gdl(q1, q2, q3, q4, q5, d6, q7, grip)

    # 1. Base pedestal y torreta
    line_pedestal.set_data([0, 0], [0, 0])
    line_pedestal.set_3d_properties([0, geom['p_base'][2]])

    line_turret.set_data([0, 0], [0, 0])
    line_turret.set_3d_properties([geom['p_base'][2], geom['p_shoulder'][2]])

    # 2. Brazo naranja J1
    p_sh = geom['p_shoulder']
    p_el = geom['p_elbow']
    line_arm1.set_data([p_sh[0], p_el[0]], [p_sh[1], p_el[1]])
    line_arm1.set_3d_properties([p_sh[2], p_el[2]])

    # 3. Antebrazo gris J2
    p_hd = geom['p_head']
    line_arm2.set_data([p_el[0], p_hd[0]], [p_el[1], p_hd[1]])
    line_arm2.set_3d_properties([p_el[2], p_hd[2]])

    # 4. Bloque guía vertical
    p_gt = geom['p_guide_top']
    p_gb = geom['p_guide_bot']
    line_guide.set_data([p_gt[0], p_gb[0]], [p_gt[1], p_gb[1]])
    line_guide.set_3d_properties([p_gt[2], p_gb[2]])

    # 5. Barra cromada vertical
    p_rt = geom['p_rod_top']
    p_rb = geom['p_rod_bot']
    line_rod.set_data([p_rt[0], p_rb[0]], [p_rt[1], p_rb[1]])
    line_rod.set_3d_properties([p_rt[2], p_rb[2]])

    # Motor superior Z
    p_mt = geom['p_motor']
    line_motor.set_data([p_rt[0], p_mt[0]], [p_rt[1], p_mt[1]])
    line_motor.set_3d_properties([p_rt[2], p_mt[2]])

    # 6. Muñeca WRIST 4
    p_wr = geom['p_wrist']
    line_wrist.set_data([p_rb[0], p_wr[0]], [p_rb[1], p_wr[1]])
    line_wrist.set_3d_properties([p_rb[2], p_wr[2]])

    # 7. Pinza de agarre
    f1 = geom['f1_nodes']
    f2 = geom['f2_nodes']
    line_f1.set_data(f1[:, 0], f1[:, 1])
    line_f1.set_3d_properties(f1[:, 2])

    line_f2.set_data(f2[:, 0], f2[:, 1])
    line_f2.set_3d_properties(f2[:, 2])

    # Articulaciones (puntos esféricos)
    joint_pts = np.array([p_sh, p_el, p_hd, p_wr])
    scat_joints._offsets3d = (joint_pts[:, 0], joint_pts[:, 1], joint_pts[:, 2])

    # Tríada de orientación del efector final
    ee_o = geom['ee_pos']
    line_ee_x.set_data([ee_o[0], geom['ee_x'][0]], [ee_o[1], geom['ee_x'][1]])
    line_ee_x.set_3d_properties([ee_o[2], geom['ee_x'][2]])

    line_ee_y.set_data([ee_o[0], geom['ee_y'][0]], [ee_o[1], geom['ee_y'][1]])
    line_ee_y.set_3d_properties([ee_o[2], geom['ee_y'][2]])

    line_ee_z.set_data([ee_o[0], geom['ee_z'][0]], [ee_o[1], geom['ee_z'][1]])
    line_ee_z.set_3d_properties([ee_o[2], geom['ee_z'][2]])

    # Actualizar texto HUD
    r_ee, p_ee, y_ee = geom['ee_rpy']
    info_str = (
        f"  ESTADO CINEMÁTICO (7 GDL + Pinza)\n"
        f"  ──────────────────────────────────\n"
        f"  Posición TCP : X:{ee_o[0]:5.1f}  Y:{ee_o[1]:5.1f}  Z:{ee_o[2]:5.1f} cm\n"
        f"  Orient. RPY  : R:{r_ee:5.1f}°  P:{p_ee:5.1f}°  Y:{y_ee:5.1f}°\n"
        f"  Carrera Z d6 : {d6:5.1f} cm  |  Pinza : {grip:5.1f} cm"
    )
    info_text.set_text(info_str)

    fig.canvas.draw_idle()

# ==============================================================================
# 5. CONTROLES INTERACTIVOS (SLIDERS Y BOTONES)
# ==============================================================================

# Disposición en 2 columnas para 8 sliders + botones
y_base = 0.075
h_slider = 0.028
gap_y = 0.048

# Columna 1 (Articulaciones Base y Brazo)
ax_q1 = plt.axes([0.14, y_base + 3 * gap_y, 0.32, h_slider], facecolor='#263238')
ax_q2 = plt.axes([0.14, y_base + 2 * gap_y, 0.32, h_slider], facecolor='#263238')
ax_q3 = plt.axes([0.14, y_base + 1 * gap_y, 0.32, h_slider], facecolor='#263238')
ax_q4 = plt.axes([0.14, y_base + 0 * gap_y, 0.32, h_slider], facecolor='#263238')

# Columna 2 (Cabeza, Eje Z prismático, Muñeca y Pinza)
ax_q5   = plt.axes([0.62, y_base + 3 * gap_y, 0.32, h_slider], facecolor='#263238')
ax_d6   = plt.axes([0.62, y_base + 2 * gap_y, 0.32, h_slider], facecolor='#263238')
ax_q7   = plt.axes([0.62, y_base + 1 * gap_y, 0.32, h_slider], facecolor='#263238')
ax_grip = plt.axes([0.62, y_base + 0 * gap_y, 0.32, h_slider], facecolor='#263238')

# Sliders con colores y rangos apropiados
s_q1 = Slider(ax_q1, 'q1 (Base Yaw)', -180.0, 180.0, valinit=0.0, valstep=1.0, color='#78909C')
s_q2 = Slider(ax_q2, 'q2 (Hombro Pitch)', -90.0, 90.0, valinit=25.0, valstep=1.0, color='#FF6F00')
s_q3 = Slider(ax_q3, 'q3 (Codo Pitch)', -150.0, 150.0, valinit=50.0, valstep=1.0, color='#FF8F00')
s_q4 = Slider(ax_q4, 'q4 (Antebrazo Roll)', -180.0, 180.0, valinit=0.0, valstep=1.0, color='#90A4AE')

s_q5 = Slider(ax_q5, 'q5 (Cabeza Pitch)', -180.0, 180.0, valinit=-75.0, valstep=1.0, color='#546E7A')
s_d6 = Slider(ax_d6, 'd6 (Prismático Z)', -15.0, 15.0, valinit=5.0, valstep=0.5, color='#ECEFF1')
s_q7 = Slider(ax_q7, 'q7 (Muñeca Roll)', -180.0, 180.0, valinit=0.0, valstep=1.0, color='#607D8B')
s_grip = Slider(ax_grip, 'Pinza (Apertura)', 0.4, 8.0, valinit=3.5, valstep=0.2, color='#FF6F00')

# Estilo de texto de los sliders
for s in [s_q1, s_q2, s_q3, s_q4, s_q5, s_d6, s_q7, s_grip]:
    s.label.set_color('#ECEFF1')
    s.label.set_fontsize(8.5)
    s.valtext.set_color('#FFFFFF')
    s.valtext.set_fontsize(8.5)

# Asignar callback de actualización
s_q1.on_changed(update)
s_q2.on_changed(update)
s_q3.on_changed(update)
s_q4.on_changed(update)
s_q5.on_changed(update)
s_d6.on_changed(update)
s_q7.on_changed(update)
s_grip.on_changed(update)

# Botones de presets y reset
ax_btn_reset = plt.axes([0.18, 0.015, 0.16, 0.038])
ax_btn_pose  = plt.axes([0.38, 0.015, 0.22, 0.038])
ax_btn_grip  = plt.axes([0.64, 0.015, 0.20, 0.038])

btn_reset = Button(ax_btn_reset, 'Posición cero', color='#37474F', hovercolor='#455A64')
btn_pose  = Button(ax_btn_pose,  'Pose inicial', color='#E65100', hovercolor='#FF6F00')
btn_grip  = Button(ax_btn_grip,  'Abrir/Cerrar Pinza', color='#263238', hovercolor='#37474F')

btn_reset.label.set_color('#FFFFFF')
btn_reset.label.set_fontsize(8.5)
btn_pose.label.set_color('#FFFFFF')
btn_pose.label.set_fontsize(8.5)
btn_pose.label.set_weight('bold')
btn_grip.label.set_color('#FFFFFF')
btn_grip.label.set_fontsize(8.5)

def on_reset_clicked(event):
    s_q1.reset()
    s_q2.set_val(0.0)
    s_q3.set_val(0.0)
    s_q4.set_val(0.0)
    s_q5.set_val(0.0)
    s_d6.set_val(0.0)
    s_q7.set_val(0.0)
    s_grip.set_val(3.5)

def on_pose_clicked(event):
    s_q1.set_val(0.0)
    s_q2.set_val(25.0)
    s_q3.set_val(50.0)
    s_q4.set_val(0.0)
    s_q5.set_val(-75.0)
    s_d6.set_val(5.0)
    s_q7.set_val(0.0)
    s_grip.set_val(3.5)

is_gripper_open = [True]
def on_grip_toggle_clicked(event):
    if is_gripper_open[0]:
        s_grip.set_val(0.6) # Cerrar pinza
        is_gripper_open[0] = False
    else:
        s_grip.set_val(6.0) # Abrir pinza
        is_gripper_open[0] = True

btn_reset.on_clicked(on_reset_clicked)
btn_pose.on_clicked(on_pose_clicked)
btn_grip.on_clicked(on_grip_toggle_clicked)

# Dibujado inicial
update(0)

if __name__ == '__main__':
    plt.show()
