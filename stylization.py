import subprocess
import os


# Call Ebsynth to stylize the frame. Style, Source, and all Guides should be strings leading to the file paths.
def stylize_frame(style_file, source_file, g_col_file, edge_source_file1, edge_g_col_file1, flow_source_file, flow_g_col_file, output_file, stylized_image_file=None, o_hat_i_file=None):
    ebsynth_exec = "./ebsynth.exe"
    cmd = [
        ebsynth_exec,
        "-style", style_file,  # S - style exemplar
        "-guide", source_file, g_col_file, # Gcol - style exemplar as the color guide
        "-weight", "5.0",
    ]
    # Gmask isn't needed.
    if edge_source_file1 is not None and edge_g_col_file1 is not None:
        cmd.extend(["-guide", edge_source_file1, edge_g_col_file1, "-weight", "0.5"])  # Gedge - edge guide for target and source frames
    if flow_source_file is not None and flow_g_col_file is not None:
        cmd.extend(["-guide", flow_source_file, flow_g_col_file, "-weight", "2.0"])  # Gpos - positional guide using optical flow
    if stylized_image_file is not None and o_hat_i_file is not None:
        cmd.extend(["-guide", stylized_image_file, o_hat_i_file, "-weight", "0.5"])  # Gtemp - temporal guide using previous stylized frame
    cmd.extend([
        "-output", output_file,
        "-searchvoteiters", "12",
        "-patchmatchiters", "6",
        "-pyramidlevels", "6",
        "-patchsize", "5",

    ])

    print(" ".join(cmd))
    subprocess.call(cmd)
    return output_file