import os
import glob
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
from pygments.lexers import get_lexer_for_filename, guess_lexer

from pygments.styles import get_style_by_name
import difflib

from moviepy import VideoClip
import numpy as np
from shinka.utils import load_programs_to_df, get_path_to_best_node, store_best_path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_dir",
    type=str,
    default="examples/agent_design/results_20250620_133347",
)
args = parser.parse_args()

results_dir = args.results_dir

df = load_programs_to_df(f"{results_dir}/evolution_db.sqlite")
print(df.head())
best_path = get_path_to_best_node(df)
store_best_path(best_path, results_dir)
best_path_dir = os.path.join(results_dir, "best_path")

                             
shutil.copy(f"{best_path_dir}/main.py", f"{best_path_dir}/original.py")
                       
BASE_CODE_FILE = f"{best_path_dir}/original.py"                               
PATCH_DIR = f"{best_path_dir}/patches"                                          

                                                                          
                                                                        
                             
INIT_LABEL_STATE = "Initial"
patch_labels = best_path.patch_name.iloc[1:].tolist()


OUTPUT_VIDEO = f"{results_dir}/code_evolution.mp4"
                      
                      
VIDEO_WIDTH = 3840                               
VIDEO_HEIGHT = 2160                               
FPS = 25

                         
CENTER_CONTENT = True                                          

                    
USE_WHITE_BACKGROUND = (
    True                                                               
)

            
                                                         
FONT_PATH = (
    "/System/Library/Fonts/Menlo.ttc"           
    if os.path.exists("/System/Library/Fonts/Menlo.ttc")
    else "/System/Library/Fonts/SFMono-Regular.otf"           
    if os.path.exists("/System/Library/Fonts/SFMono-Regular.otf")
    else "/System/Library/Fonts/Monaco.ttf"          
    if os.path.exists("/System/Library/Fonts/Monaco.ttf")
    else "/System/Library/Fonts/Courier.ttc"           
    if os.path.exists("/System/Library/Fonts/Courier.ttc")
    else "/System/Library/Fonts/Andale Mono.ttf"               
    if os.path.exists("/System/Library/Fonts/Andale Mono.ttf")
    else "/System/Library/Fonts/Menlo.ttc"                     
)

FONT_PATH = "/System/Library/Fonts/Andale Mono.ttf"
try:
    FONT_SIZE = 30
    LINE_HEIGHT_RATIO = 1.3
    LINE_HEIGHT = int(FONT_SIZE * LINE_HEIGHT_RATIO)
    FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print(f"Warning: Font {FONT_PATH} not found. Using PIL default font.")
    FONT = ImageFont.load_default()
    try:
        bbox = FONT.getbbox("M")
        FONT_SIZE = bbox[3] - bbox[1] if bbox else 10
        LINE_HEIGHT_RATIO = 1.5
        LINE_HEIGHT = int(FONT_SIZE * LINE_HEIGHT_RATIO)
    except AttributeError:
        FONT_SIZE = 10
        LINE_HEIGHT_RATIO = 1.5
        LINE_HEIGHT = 15
        print("Using fallback font size and line height.")


                                               
if USE_WHITE_BACKGROUND:
                        
    BG_COLOR = (250, 250, 250)
    TEXT_COLOR = (30, 30, 30)
    ADDED_LINE_BG_COLOR = (200, 255, 200, 180)                          
    HISTORY_ADDED_LINE_BG_COLOR = (180, 240, 180, 160)
    HISTORY_PANE_BG_COLOR = (240, 240, 240)
    HISTORY_PANE_BORDER_COLOR = (180, 180, 180)
    HISTORY_PANE_TEXT_COLOR = (30, 30, 30)
    MINI_DIFF_BG_COLOR = (230, 230, 230)
    MINI_DIFF_ADDED_COLOR = (0, 120, 0)
    MINI_DIFF_REMOVED_COLOR = (150, 0, 0)
    MINI_DIFF_CONTEXT_COLOR = (120, 120, 120)
    MINI_DIFF_ACTIVE_BORDER_COLOR = (180, 160, 0)
    PYGMENTS_STYLE = "default"                             
else:
                                 
    BG_COLOR = (30, 30, 30)
    TEXT_COLOR = (220, 220, 220)
    ADDED_LINE_BG_COLOR = (30, 70, 30, 200)                         
    HISTORY_ADDED_LINE_BG_COLOR = (25, 55, 25, 180)
    HISTORY_PANE_BG_COLOR = (38, 38, 38)
    HISTORY_PANE_BORDER_COLOR = (60, 60, 60)
    HISTORY_PANE_TEXT_COLOR = (200, 200, 200)
    MINI_DIFF_BG_COLOR = (45, 45, 45)
    MINI_DIFF_ADDED_COLOR = (0, 180, 0)
    MINI_DIFF_REMOVED_COLOR = (180, 0, 0)
    MINI_DIFF_CONTEXT_COLOR = (100, 100, 100)
    MINI_DIFF_ACTIVE_BORDER_COLOR = (220, 220, 0)
    PYGMENTS_STYLE = "monokai"                            

style = get_style_by_name(PYGMENTS_STYLE)

                                          
SHOW_HISTORY_PANES = True
NUM_HISTORY_PANES_TO_SHOW = 3
HISTORY_PANE_Y_START = 94
HISTORY_PANE_X_START_OFFSET_FROM_RIGHT = 5
HISTORY_PANE_SPACING = 10
HISTORY_FONT_SIZE = 8
HISTORY_LABEL_FONT_SIZE = 30                                            
TITLE_FONT_SIZE = 55                                          
HISTORY_LINE_HEIGHT_RATIO = 1.2
HISTORY_MAX_LINES_TO_DRAW = 1000
MAIN_PANE_X_OFFSET = 20
                                        
MAIN_PANE_Y_OFFSET = HISTORY_PANE_Y_START
MAIN_PANE_RIGHT_MARGIN_IF_HISTORY = 5

                                                                      
MINI_DIFF_PANE_WIDTH = 150
MINI_DIFF_WIDTH = 12
MINI_DIFF_HEIGHT_PER_LINE = 2
MINI_DIFF_SPACING = 7
MINI_DIFF_MAX_LINES = 1000
MINI_DIFF_TEXT_SIZE = 30


                  
CHARS_PER_SECOND = 150
HOLD_DURATION_PER_ITERATION = (
    1.0                                                          
)
                                                            
SCROLL_SPEED_LINES_PER_SECOND = 40                              
MIN_SCROLL_DURATION = 0.75                                       
SCROLL_PAUSE_AT_TOP = 0.1                                 
SCROLL_PAUSE_AT_BOTTOM = 1.5                                         

                                               
HISTORY_TRANSITION_DURATION = 0.8                                              
                                              
MAIN_PANE_SLIDE_IN_DURATION = 0.6

                          


def get_file_content(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def apply_patch(base_file_path, patch_file_path, target_dir):
    temp_file_name = os.path.basename(base_file_path)
    temp_file_path = os.path.join(target_dir, temp_file_name)

    if not os.path.exists(temp_file_path) and os.path.exists(base_file_path):
        shutil.copy2(base_file_path, temp_file_path)
    elif not os.path.exists(temp_file_path) and not os.path.exists(base_file_path):
        with open(temp_file_path, "w", encoding="utf-8") as f:                
            pass

    cmd = [
        "git",
        "apply",
        "--ignore-whitespace",
        "--recount",
    ]                                     
                              
    patch_preview_content = get_file_content(patch_file_path)
    if (
        f"--- a/{temp_file_name}" in patch_preview_content
        or f"--- a\\{temp_file_name}" in patch_preview_content
    ):
        cmd.append("-p1")

    cmd.append(os.path.abspath(patch_file_path))

    try:
        subprocess.run(
            cmd,
            cwd=target_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except subprocess.CalledProcessError as e:
        print(f"Error applying patch {patch_file_path} with git apply:")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        print("Attempting manual patch application (VERY basic)...")
                                                                       
        current_content_lines = get_file_content(temp_file_path).splitlines(True)
        patch_content_lines = patch_preview_content.splitlines(True)

        output_lines = []
        c_idx = 0
        for p_line in patch_content_lines:
            if (
                p_line.startswith("---")
                or p_line.startswith("+++")
                or p_line.startswith("@@")
                or p_line.startswith("diff")
            ):
                continue
            if p_line.startswith("+"):
                output_lines.append(p_line[1:])
            elif p_line.startswith("-"):
                c_idx += 1                                      
            else:           
                if c_idx < len(current_content_lines):
                    output_lines.append(current_content_lines[c_idx])
                c_idx += 1
        if c_idx < len(current_content_lines):                    
            output_lines.extend(current_content_lines[c_idx:])

        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.writelines(output_lines)
        print(
            f"Manual patch attempt for {patch_file_path} completed. "
            f"Result may be imperfect."
        )

    return get_file_content(temp_file_path)


def get_diff_details(old_code, new_code):
    added_lines_indices = set()
    new_code_lines = new_code.splitlines()
    old_code_lines = old_code.splitlines()

    s = difflib.SequenceMatcher(None, old_code_lines, new_code_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "insert":
            for i in range(j1, j2):
                added_lines_indices.add(i)
        elif tag == "replace":
            for i in range(j1, j2):
                added_lines_indices.add(i)
    return added_lines_indices


def draw_mini_diff(
    patch_content, is_active=False, font_mini_diff=None
):                     
    img_height = MINI_DIFF_HEIGHT_PER_LINE * MINI_DIFF_MAX_LINES
    img_width = MINI_DIFF_WIDTH

    final_img_width = img_width + (4 if is_active else 0)
    final_img_height = img_height + (4 if is_active else 0)

    base_img = Image.new("RGB", (img_width, img_height), MINI_DIFF_BG_COLOR)
    draw = ImageDraw.Draw(base_img)
    y = 0
    lines_drawn = 0
    for line in patch_content.splitlines():
        if lines_drawn >= MINI_DIFF_MAX_LINES:
            break
        if (
            line.startswith("+++")
            or line.startswith("---")
            or line.startswith("diff")
            or line.startswith("index")
            or line.startswith("@@")
        ):
            continue
        color = MINI_DIFF_CONTEXT_COLOR
        if line.startswith("+"):
            color = MINI_DIFF_ADDED_COLOR
        elif line.startswith("-"):
            color = MINI_DIFF_REMOVED_COLOR
        draw.rectangle((0, y, img_width, y + MINI_DIFF_HEIGHT_PER_LINE), fill=color)
        y += MINI_DIFF_HEIGHT_PER_LINE
        lines_drawn += 1

    if is_active:
        border_img = Image.new(
            "RGB", (final_img_width, final_img_height), MINI_DIFF_ACTIVE_BORDER_COLOR
        )
        border_img.paste(base_img, (2, 2))
        return border_img
    return base_img


                       
try:
    HISTORY_FONT = ImageFont.truetype(FONT_PATH, HISTORY_FONT_SIZE)
    HISTORY_LABEL_FONT = ImageFont.truetype(FONT_PATH, HISTORY_LABEL_FONT_SIZE)
    TITLE_FONT = ImageFont.truetype(FONT_PATH, TITLE_FONT_SIZE)
except IOError:
    HISTORY_FONT = FONT            
    HISTORY_LABEL_FONT = FONT            
    TITLE_FONT = FONT                           
HISTORY_LINE_HEIGHT = int(HISTORY_FONT_SIZE * HISTORY_LINE_HEIGHT_RATIO)

try:
    MINI_DIFF_FONT = ImageFont.truetype(FONT_PATH, MINI_DIFF_TEXT_SIZE)
except IOError:
    MINI_DIFF_FONT = ImageFont.load_default()


                        
print("Preparing code states and diffs...")
patch_files = sorted(glob.glob(os.path.join(PATCH_DIR, "*.patch")))
if not patch_files:
    print(f"No patch files found in {PATCH_DIR}. Exiting.")
    exit()

if not os.path.exists(BASE_CODE_FILE):
    print(f"Base code file {BASE_CODE_FILE} not found. Assuming empty base.")
    with open(BASE_CODE_FILE, "w", encoding="utf-8") as f:
        pass

code_states = []
raw_patch_contents_for_minidiff = []                                

base_content = get_file_content(BASE_CODE_FILE)
code_states.append(
    {"content": base_content, "added_lines": set(), "patch_name": INIT_LABEL_STATE}
)

temp_dir = tempfile.mkdtemp()
temp_base_path = os.path.join(temp_dir, os.path.basename(BASE_CODE_FILE))
if os.path.exists(BASE_CODE_FILE):
    shutil.copy2(BASE_CODE_FILE, temp_base_path)
else:
    with open(temp_base_path, "w", encoding="utf-8") as f:
        pass

previous_content = base_content
for i, patch_file in enumerate(patch_files):
    print(f"Processing patch {i + 1}/{len(patch_files)}: {patch_file}")
    patch_name = os.path.basename(patch_file)

                                                                 
    if patch_labels and i < len(patch_labels):
        display_name = patch_labels[i]
    else:
        display_name = patch_name

    current_content = apply_patch(temp_base_path, patch_file, temp_dir)

    added_lines_indices = get_diff_details(previous_content, current_content)
    code_states.append(
        {
            "content": current_content,
            "added_lines": added_lines_indices,
            "patch_name": display_name,
        }
    )

                                                    
    if not SHOW_HISTORY_PANES:
        raw_patch_contents_for_minidiff.append(get_file_content(patch_file))

    previous_content = current_content

                                                                        
mini_diff_images = []
if not SHOW_HISTORY_PANES:
    for i, raw_patch_text in enumerate(raw_patch_contents_for_minidiff):
        mini_diff_images.append(
            draw_mini_diff(
                raw_patch_text, is_active=False, font_mini_diff=MINI_DIFF_FONT
            )
        )


                          
total_duration = 0
for i, state in enumerate(code_states):
    lines_in_state = len(state["content"].splitlines())
    max_visible_lines = (VIDEO_HEIGHT - MAIN_PANE_Y_OFFSET - 50) // LINE_HEIGHT

                                                                  
                                                                    
    if SHOW_HISTORY_PANES and i > 0:
                                                       
        estimated_main_pane_width = int(VIDEO_WIDTH * 0.4)
    else:
                                                  
        estimated_main_pane_width = VIDEO_WIDTH - MAIN_PANE_X_OFFSET * 2

    scroll_duration_for_state = 0
    if lines_in_state > max_visible_lines:
        lines_to_scroll = lines_in_state - max_visible_lines
        scroll_duration_for_state = max(
            MIN_SCROLL_DURATION, lines_to_scroll / SCROLL_SPEED_LINES_PER_SECOND
        )
                                       
        duration_for_this_state = (
            HOLD_DURATION_PER_ITERATION
            + scroll_duration_for_state
            + SCROLL_PAUSE_AT_TOP
            + SCROLL_PAUSE_AT_BOTTOM
        )
    else:
                             
        duration_for_this_state = HOLD_DURATION_PER_ITERATION

                                                          
    if i > 0:                                                         
        duration_for_this_state += MAIN_PANE_SLIDE_IN_DURATION
    total_duration += duration_for_this_state

print(f"Total estimated duration: {total_duration:.2f}s")

try:
    lexer = get_lexer_for_filename(BASE_CODE_FILE, stripall=False)
except Exception:                     
    try:
        lexer = guess_lexer(code_states[0]["content"] if code_states else "")
    except Exception:                     
        from pygments.lexers.special import TextLexer

        lexer = TextLexer()
print(f"Using Pygments lexer: {lexer.name}")


                            
def make_frame(t):
                                                                        
    current_time = 0
    state_index = 0
    scroll_offset = 0
    history_transition_alpha = (
        1.0                                                                   
    )
    main_pane_transition_progress = (
        1.0                                                      
    )
    main_pane_slide_progress = 1.0                                              

    for i, state in enumerate(code_states):
        lines_in_state = len(state["content"].splitlines())
        max_visible_lines = (VIDEO_HEIGHT - MAIN_PANE_Y_OFFSET - 50) // LINE_HEIGHT

                                                                      
                                                                        
        if SHOW_HISTORY_PANES and i > 0:
                                                           
            estimated_main_pane_width = int(VIDEO_WIDTH * 0.4)
        else:
                                                      
            estimated_main_pane_width = VIDEO_WIDTH - MAIN_PANE_X_OFFSET * 2

        scroll_duration_for_state = 0
        if lines_in_state > max_visible_lines:
            lines_to_scroll = lines_in_state - max_visible_lines
            scroll_duration_for_state = max(
                MIN_SCROLL_DURATION, lines_to_scroll / SCROLL_SPEED_LINES_PER_SECOND
            )
                                        
            base_state_duration = (
                HOLD_DURATION_PER_ITERATION
                + scroll_duration_for_state
                + SCROLL_PAUSE_AT_TOP
                + SCROLL_PAUSE_AT_BOTTOM
            )
        else:
                                                
            base_state_duration = HOLD_DURATION_PER_ITERATION

                                                              
        if i > 0:                                                         
            state_duration = base_state_duration + MAIN_PANE_SLIDE_IN_DURATION
        else:
            state_duration = base_state_duration

        if t >= current_time and t < current_time + state_duration:
            state_index = i
            time_in_state = t - current_time

                                                                             
            if i > 0 and time_in_state < MAIN_PANE_SLIDE_IN_DURATION:
                                                       
                main_pane_slide_progress = time_in_state / MAIN_PANE_SLIDE_IN_DURATION
                                                                     
                            
                scroll_offset = 0
                content_time_progress = 0                                     
            else:
                                                                          
                main_pane_slide_progress = 1.0
                if i > 0:
                    content_time_progress = time_in_state - MAIN_PANE_SLIDE_IN_DURATION
                else:
                    content_time_progress = time_in_state

                                                             
                                      
            if i == 0:
                                                     
                history_transition_alpha = 0.0
                main_pane_transition_progress = 0.0
            else:
                                                                     
                history_transition_alpha = 1.0
                main_pane_transition_progress = 1.0

                                                                   
            if lines_in_state > max_visible_lines and content_time_progress >= 0:
                if content_time_progress < HOLD_DURATION_PER_ITERATION:
                                         
                    scroll_offset = 0
                elif (
                    content_time_progress
                    < HOLD_DURATION_PER_ITERATION + SCROLL_PAUSE_AT_TOP
                ):
                                                   
                    scroll_offset = 0
                elif (
                    content_time_progress
                    < HOLD_DURATION_PER_ITERATION
                    + SCROLL_PAUSE_AT_TOP
                    + scroll_duration_for_state
                ):
                                     
                    scroll_progress = (
                        content_time_progress
                        - HOLD_DURATION_PER_ITERATION
                        - SCROLL_PAUSE_AT_TOP
                    ) / scroll_duration_for_state

                                                                                           
                                                                                             
                    state_lines = state["content"].splitlines()
                    available_height = VIDEO_HEIGHT - MAIN_PANE_Y_OFFSET - 50
                    max_width = estimated_main_pane_width - 20

                                                                                         
                    current_height = 0
                    optimal_scroll_offset = len(state_lines)

                    for line_idx in range(len(state_lines) - 1, -1, -1):
                        line_text = state_lines[line_idx].rstrip("\r\n")

                                                                                
                        if not line_text:
                            line_height = LINE_HEIGHT
                        else:
                                                       
                            line_width = len(line_text) * (
                                FONT_SIZE * 0.6
                            )                  
                            if line_width > max_width:
                                                                                            
                                estimated_wrap_lines = max(
                                    1, int(line_width / max_width) + 1
                                )
                                line_height = LINE_HEIGHT * estimated_wrap_lines
                            else:
                                line_height = LINE_HEIGHT

                        if current_height + line_height > available_height:
                            optimal_scroll_offset = line_idx + 1
                            break
                        current_height += line_height

                    optimal_scroll_offset = max(0, optimal_scroll_offset)
                    scroll_offset = int(scroll_progress * optimal_scroll_offset)
                else:
                                                                         
                    state_lines = state["content"].splitlines()
                    available_height = VIDEO_HEIGHT - MAIN_PANE_Y_OFFSET - 50
                    max_width = estimated_main_pane_width - 20

                    current_height = 0
                    optimal_scroll_offset = len(state_lines)

                    for line_idx in range(len(state_lines) - 1, -1, -1):
                        line_text = state_lines[line_idx].rstrip("\r\n")

                        if not line_text:
                            line_height = LINE_HEIGHT
                        else:
                            line_width = len(line_text) * (FONT_SIZE * 0.6)
                            if line_width > max_width:
                                estimated_wrap_lines = max(
                                    1, int(line_width / max_width) + 1
                                )
                                line_height = LINE_HEIGHT * estimated_wrap_lines
                            else:
                                line_height = LINE_HEIGHT

                        if current_height + line_height > available_height:
                            optimal_scroll_offset = line_idx + 1
                            break
                        current_height += line_height

                    scroll_offset = max(0, optimal_scroll_offset)
            else:
                scroll_offset = 0
            break
        current_time += state_duration

    if state_index >= len(code_states):
        state_index = len(code_states) - 1

    current_state_data = code_states[state_index]
    code_to_display_full = current_state_data["content"]
    code_to_display_typed = code_to_display_full                            
    added_lines_for_this_main_state = current_state_data["added_lines"]
    iter_idx = state_index                                       

    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")

                                                               
    actual_num_history_panes_to_render = 0
    history_pane_individual_width = 0

                                                                       
    main_code_pane_width_no_history = VIDEO_WIDTH - MAIN_PANE_X_OFFSET * 2
    main_pane_x_offset_no_history = MAIN_PANE_X_OFFSET

    main_code_pane_width_with_history = main_code_pane_width_no_history
    main_pane_x_offset_with_history = MAIN_PANE_X_OFFSET
    total_width_for_all_history_panes = 0

    if SHOW_HISTORY_PANES and iter_idx > 0:
        actual_num_history_panes_to_render = min(NUM_HISTORY_PANES_TO_SHOW, iter_idx)

                                                 
        total_history_spacing = (
            (actual_num_history_panes_to_render - 1) * HISTORY_PANE_SPACING
            if actual_num_history_panes_to_render > 0
            else 0
        )
                                                                
                                         
                                                                       
        total_width_for_all_history_panes = (
            int(VIDEO_WIDTH * 0.6) - HISTORY_PANE_X_START_OFFSET_FROM_RIGHT
        )

        if actual_num_history_panes_to_render > 0:
            history_pane_individual_width = int(
                (total_width_for_all_history_panes - total_history_spacing)
                / actual_num_history_panes_to_render
            )
            history_pane_individual_width = max(
                100, history_pane_individual_width
            )             

            main_code_pane_width_with_history = VIDEO_WIDTH - (
                total_width_for_all_history_panes
                + MAIN_PANE_X_OFFSET
                + MAIN_PANE_RIGHT_MARGIN_IF_HISTORY
                + HISTORY_PANE_X_START_OFFSET_FROM_RIGHT
            )
            main_code_pane_width_with_history = max(
                int(VIDEO_WIDTH * 0.35), main_code_pane_width_with_history
            )                                          
    elif not SHOW_HISTORY_PANES:
        main_code_pane_width_no_history = (
            VIDEO_WIDTH - MAIN_PANE_X_OFFSET - MINI_DIFF_PANE_WIDTH - 10
        )                       

                                                             
    main_code_pane_width = int(
        main_code_pane_width_no_history * (1 - main_pane_transition_progress)
        + main_code_pane_width_with_history * main_pane_transition_progress
    )

                                                
    if CENTER_CONTENT:
        content_total_width_no_history = main_code_pane_width_no_history
        content_total_width_with_history = main_code_pane_width_with_history
        if SHOW_HISTORY_PANES and actual_num_history_panes_to_render > 0:
            content_total_width_with_history += (
                total_width_for_all_history_panes
                + HISTORY_PANE_X_START_OFFSET_FROM_RIGHT
                + MAIN_PANE_RIGHT_MARGIN_IF_HISTORY
            )

        main_pane_x_offset_no_history = max(
            MAIN_PANE_X_OFFSET, (VIDEO_WIDTH - content_total_width_no_history) // 2
        )
        main_pane_x_offset_with_history = max(
            MAIN_PANE_X_OFFSET, (VIDEO_WIDTH - content_total_width_with_history) // 2
        )

                                                            
    main_pane_x_offset_final = int(
        main_pane_x_offset_no_history * (1 - main_pane_transition_progress)
        + main_pane_x_offset_with_history * main_pane_transition_progress
    )

                                                           
    slide_in_start_x = -main_code_pane_width                                           
    main_pane_x_offset = int(
        slide_in_start_x * (1 - main_pane_slide_progress)
        + main_pane_x_offset_final * main_pane_slide_progress
    )

                                                      
    iter_text_content = f"{current_state_data['patch_name']}"
    title_margin_top = 67                                       
    title_padding = 12                                   

                                       
    iter_text_w = (
        TITLE_FONT.getlength(iter_text_content)
        if hasattr(TITLE_FONT, "getlength")
        else draw.textlength(iter_text_content, font=TITLE_FONT)
        if hasattr(draw, "textlength")
        else TITLE_FONT.getbbox(iter_text_content)[2]
        if hasattr(TITLE_FONT, "getbbox")
        else 400
    )

                                         
    title_center_x = main_pane_x_offset + (main_code_pane_width // 2)
    title_x_start = title_center_x - (iter_text_w // 2) - title_padding
    title_x_end = title_center_x + (iter_text_w // 2) + title_padding
    title_text_x = title_center_x - (iter_text_w // 2)

                                                     
                                                                       
    title_bg_color = (60, 60, 60) if not USE_WHITE_BACKGROUND else (220, 220, 220)

    title_box_y1 = MAIN_PANE_Y_OFFSET - title_margin_top - title_padding
    title_box_y2 = (
        MAIN_PANE_Y_OFFSET - title_margin_top + TITLE_FONT_SIZE + title_padding
    )

    draw.rectangle(
        (
            title_x_start,
            title_box_y1,
            title_x_end,
            title_box_y2,
        ),
        fill=title_bg_color,
    )

               
    draw.text(
        (title_text_x, MAIN_PANE_Y_OFFSET - title_margin_top),
        iter_text_content,
        font=TITLE_FONT,
        fill=TEXT_COLOR,
    )

                                                  
    y_offset = MAIN_PANE_Y_OFFSET
    main_lines = code_to_display_typed.splitlines(True)

                                                       
    total_code_lines = len(main_lines)
    max_displayable_lines = min(
        total_code_lines, (VIDEO_HEIGHT - MAIN_PANE_Y_OFFSET - 50) // LINE_HEIGHT
    )
                      
    main_pane_height = max_displayable_lines * LINE_HEIGHT + 20

                            
    start_line = scroll_offset
    end_line = min(start_line + max_displayable_lines, total_code_lines)

    for line_idx in range(start_line, end_line):
        if line_idx >= len(main_lines):
            break
        if y_offset + LINE_HEIGHT > MAIN_PANE_Y_OFFSET + main_pane_height:
            break

        line_text_orig = main_lines[line_idx]
        line_text = line_text_orig.rstrip("\r\n")

                                               
        max_width = main_code_pane_width - 20
        line_width = (
            FONT.getlength(line_text)
            if hasattr(FONT, "getlength")
            else draw.textlength(line_text, font=FONT)
            if hasattr(draw, "textlength")
            else FONT.getbbox(line_text)[2]
            if hasattr(FONT, "getbbox")
            else len(line_text) * 8
        )

        if line_width > max_width:
                                                       
            avg_char_w = (
                FONT.getlength("M")
                if hasattr(FONT, "getlength")
                else draw.textlength("M", font=FONT)
                if hasattr(draw, "textlength")
                else FONT.getbbox("M")[2]
                if hasattr(FONT, "getbbox")
                else 8
            )
            chars_per_line = (
                int(max_width / avg_char_w) if avg_char_w > 0 else int(max_width / 8)
            )

                                       
            lines_to_draw = []
            remaining = line_text

            while remaining:
                if len(remaining) <= chars_per_line:
                    lines_to_draw.append(remaining)
                    break

                                                     
                split_pos = chars_per_line
                if " " in remaining[:chars_per_line]:
                                                              
                    last_space = remaining[:chars_per_line].rstrip().rfind(" ")
                    if last_space > 0:
                        split_pos = last_space + 1

                lines_to_draw.append(remaining[:split_pos])
                remaining = remaining[split_pos:]

                                
            first_line = True
            for wrapped_line in lines_to_draw:
                                                                              
                if line_idx in added_lines_for_this_main_state:
                    draw.rectangle(
                        (
                            main_pane_x_offset - 5,
                            y_offset - 2,
                            main_pane_x_offset + main_code_pane_width - 10,
                            y_offset + LINE_HEIGHT - 2,
                        ),
                        fill=ADDED_LINE_BG_COLOR,
                    )

                line_x_cursor = main_pane_x_offset
                if not first_line:
                                                       
                    line_x_cursor += 20

                try:
                    tokens_on_line = lexer.get_tokens(wrapped_line)
                    for ttype, tvalue in tokens_on_line:
                        style_for_token = style.style_for_token(ttype)
                        color = style_for_token["color"]
                        token_color = TEXT_COLOR
                        if color:
                            try:
                                token_color = (
                                    int(color[0:2], 16),
                                    int(color[2:4], 16),
                                    int(color[4:6], 16),
                                )
                            except ValueError:
                                pass

                        draw.text(
                            (line_x_cursor, y_offset),
                            tvalue,
                            font=FONT,
                            fill=token_color,
                        )

                        token_width = (
                            FONT.getlength(tvalue)
                            if hasattr(FONT, "getlength")
                            else draw.textlength(tvalue, font=FONT)
                            if hasattr(draw, "textlength")
                            else FONT.getbbox(tvalue)[2]
                            if hasattr(FONT, "getbbox")
                            else len(tvalue) * 8
                        )
                        line_x_cursor += token_width
                except Exception:
                                                                               
                    draw.text(
                        (
                            main_pane_x_offset
                            if first_line
                            else main_pane_x_offset + 20,
                            y_offset,
                        ),
                        wrapped_line,
                        font=FONT,
                        fill=TEXT_COLOR,
                    )

                y_offset += LINE_HEIGHT
                first_line = False

                                                                   
                if y_offset + LINE_HEIGHT > MAIN_PANE_Y_OFFSET + main_pane_height:
                    break
        else:
                                                        
                                             
            if line_idx in added_lines_for_this_main_state:
                draw.rectangle(
                    (
                        main_pane_x_offset - 5,
                        y_offset - 2,
                        main_pane_x_offset + main_code_pane_width - 10,
                        y_offset + LINE_HEIGHT - 2,
                    ),
                    fill=ADDED_LINE_BG_COLOR,
                )

            line_x_cursor = main_pane_x_offset
            try:
                tokens_on_line = lexer.get_tokens(line_text)
                for ttype, tvalue in tokens_on_line:
                    style_for_token = style.style_for_token(ttype)
                    color = style_for_token["color"]
                    token_color = TEXT_COLOR
                    if color:
                        try:
                            token_color = (
                                int(color[0:2], 16),
                                int(color[2:4], 16),
                                int(color[4:6], 16),
                            )
                        except ValueError:
                            pass

                    draw.text(
                        (line_x_cursor, y_offset), tvalue, font=FONT, fill=token_color
                    )

                    token_width = (
                        FONT.getlength(tvalue)
                        if hasattr(FONT, "getlength")
                        else draw.textlength(tvalue, font=FONT)
                        if hasattr(draw, "textlength")
                        else FONT.getbbox(tvalue)[2]
                        if hasattr(FONT, "getbbox")
                        else len(tvalue) * 8
                    )
                    line_x_cursor += token_width
            except Exception:
                                                                           
                draw.text(
                    (main_pane_x_offset, y_offset),
                    line_text,
                    font=FONT,
                    fill=TEXT_COLOR,
                )
            y_offset += LINE_HEIGHT

                                                           
    if (
        SHOW_HISTORY_PANES
        and actual_num_history_panes_to_render > 0
        and history_transition_alpha > 0
    ):
                                                                           
        history_img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        history_draw = ImageDraw.Draw(history_img, "RGBA")

                                                    
        current_history_pane_x_start_coord = (
            main_pane_x_offset
            + main_code_pane_width
            + MAIN_PANE_RIGHT_MARGIN_IF_HISTORY
        )

        for i in range(actual_num_history_panes_to_render):
                                                                       
            history_iter_index_to_display = iter_idx - 1 - i

            state_to_draw = code_states[history_iter_index_to_display]
            history_code_content = state_to_draw["content"]
            added_lines_in_this_history_version = state_to_draw["added_lines"]

                                                                                 
            hist_lines = history_code_content.splitlines()
            max_hist_lines = min(len(hist_lines), HISTORY_MAX_LINES_TO_DRAW)
            history_pane_height = (
                max_hist_lines * HISTORY_LINE_HEIGHT + 30
            )               

            pane_x1 = current_history_pane_x_start_coord
            pane_y1 = HISTORY_PANE_Y_START
            pane_x2 = current_history_pane_x_start_coord + history_pane_individual_width
            pane_y2 = pane_y1 + history_pane_height                         

            history_draw.rectangle(
                (pane_x1, pane_y1, pane_x2, pane_y2), fill=HISTORY_PANE_BG_COLOR
            )
            history_draw.rectangle(
                (pane_x1, pane_y1, pane_x2, pane_y2),
                outline=HISTORY_PANE_BORDER_COLOR,
                width=1,
            )

            hist_text_x = pane_x1 + 5
            hist_text_y = pane_y1 + 5

            for line_num_in_hist, hist_line_text_orig in enumerate(hist_lines):
                if line_num_in_hist >= max_hist_lines:
                    break
                if hist_text_y + HISTORY_LINE_HEIGHT > pane_y2 - 5:
                    break

                hist_line_text = hist_line_text_orig.rstrip("\r\n")

                drawable_hist_line = hist_line_text
                max_text_width_in_pane = history_pane_individual_width - 10

                                              
                current_line_width_px = (
                    HISTORY_FONT.getlength(drawable_hist_line)
                    if hasattr(HISTORY_FONT, "getlength")
                    else draw.textlength(drawable_hist_line, font=HISTORY_FONT)
                    if hasattr(draw, "textlength")
                    else HISTORY_FONT.getbbox(drawable_hist_line)[2]
                    if hasattr(HISTORY_FONT, "getbbox")
                    else len(drawable_hist_line) * HISTORY_FONT_SIZE
                )

                                                                     
                if current_line_width_px > max_text_width_in_pane:
                                                                              
                    avg_char_w = (
                        HISTORY_FONT.getlength("M")
                        if hasattr(HISTORY_FONT, "getlength")
                        else draw.textlength("M", font=HISTORY_FONT)
                        if hasattr(draw, "textlength")
                        else HISTORY_FONT.getbbox("M")[2]
                        if hasattr(HISTORY_FONT, "getbbox")
                        else HISTORY_FONT_SIZE
                    )
                    chars_per_line = (
                        int(max_text_width_in_pane / avg_char_w)
                        if avg_char_w > 0
                        else int(max_text_width_in_pane / HISTORY_FONT_SIZE)
                    )

                                               
                    wrapped_lines = []
                    remaining = drawable_hist_line

                    while (
                        remaining and hist_text_y + HISTORY_LINE_HEIGHT <= pane_y2 - 5
                    ):
                        if len(remaining) <= chars_per_line:
                            wrapped_lines.append(remaining)
                            break

                                                             
                        split_pos = chars_per_line
                        if " " in remaining[:chars_per_line]:
                                                                      
                            last_space = remaining[:chars_per_line].rstrip().rfind(" ")
                            if last_space > 0:
                                split_pos = last_space + 1

                        wrapped_lines.append(remaining[:split_pos])
                        remaining = remaining[split_pos:]

                                        
                    for wrapped_line in wrapped_lines:
                                                                              
                                 
                        if line_num_in_hist in added_lines_in_this_history_version:
                            history_draw.rectangle(
                                (
                                    hist_text_x - 2,
                                    hist_text_y,
                                    pane_x2 - 3,
                                    hist_text_y + HISTORY_LINE_HEIGHT - 1,
                                ),
                                fill=HISTORY_ADDED_LINE_BG_COLOR,
                            )
                        hist_line_x_cursor = hist_text_x
                        try:
                            tokens_on_hist_line = lexer.get_tokens(wrapped_line)
                            for ttype, tvalue in tokens_on_hist_line:
                                style_for_token = style.style_for_token(ttype)
                                color = style_for_token["color"]
                                token_color = HISTORY_PANE_TEXT_COLOR
                                if color:
                                    try:
                                        token_color = (
                                            int(color[0:2], 16),
                                            int(color[2:4], 16),
                                            int(color[4:6], 16),
                                        )
                                    except ValueError:
                                        pass

                                history_draw.text(
                                    (hist_line_x_cursor, hist_text_y),
                                    tvalue,
                                    font=HISTORY_FONT,
                                    fill=token_color,
                                )
                                token_w = (
                                    HISTORY_FONT.getlength(tvalue)
                                    if hasattr(HISTORY_FONT, "getlength")
                                    else draw.textlength(tvalue, font=HISTORY_FONT)
                                    if hasattr(draw, "textlength")
                                    else HISTORY_FONT.getbbox(tvalue)[2]
                                    if hasattr(HISTORY_FONT, "getbbox")
                                    else len(tvalue) * HISTORY_FONT_SIZE
                                )
                                hist_line_x_cursor += token_w
                                if hist_line_x_cursor > pane_x2 - 7:
                                    break
                        except Exception:
                            history_draw.text(
                                (hist_text_x, hist_text_y),
                                wrapped_line,
                                font=HISTORY_FONT,
                                fill=HISTORY_PANE_TEXT_COLOR,
                            )
                        hist_text_y += HISTORY_LINE_HEIGHT

                                                                      
                        if hist_text_y + HISTORY_LINE_HEIGHT > pane_y2 - 5:
                            break
                else:
                                                    
                                                     
                    if line_num_in_hist in added_lines_in_this_history_version:
                        history_draw.rectangle(
                            (
                                hist_text_x - 2,
                                hist_text_y,
                                pane_x2 - 3,
                                hist_text_y + HISTORY_LINE_HEIGHT - 1,
                            ),
                            fill=HISTORY_ADDED_LINE_BG_COLOR,
                        )

                    hist_line_x_cursor = hist_text_x
                    try:
                        tokens_on_hist_line = lexer.get_tokens(drawable_hist_line)
                        for ttype, tvalue in tokens_on_hist_line:
                            style_for_token = style.style_for_token(ttype)
                            color = style_for_token["color"]
                            token_color = HISTORY_PANE_TEXT_COLOR
                            if color:
                                try:
                                    token_color = (
                                        int(color[0:2], 16),
                                        int(color[2:4], 16),
                                        int(color[4:6], 16),
                                    )
                                except ValueError:
                                    pass

                            history_draw.text(
                                (hist_line_x_cursor, hist_text_y),
                                tvalue,
                                font=HISTORY_FONT,
                                fill=token_color,
                            )
                            token_w = (
                                HISTORY_FONT.getlength(tvalue)
                                if hasattr(HISTORY_FONT, "getlength")
                                else draw.textlength(tvalue, font=HISTORY_FONT)
                                if hasattr(draw, "textlength")
                                else HISTORY_FONT.getbbox(tvalue)[2]
                                if hasattr(HISTORY_FONT, "getbbox")
                                else len(tvalue) * HISTORY_FONT_SIZE
                            )
                            hist_line_x_cursor += token_w
                            if hist_line_x_cursor > pane_x2 - 7:
                                break
                    except Exception:
                        history_draw.text(
                            (hist_text_x, hist_text_y),
                            drawable_hist_line,
                            font=HISTORY_FONT,
                            fill=HISTORY_PANE_TEXT_COLOR,
                        )
                    hist_text_y += HISTORY_LINE_HEIGHT

            history_pane_label = f"{state_to_draw['patch_name']}"
                                                       
                                             
            label_w = (
                HISTORY_LABEL_FONT.getlength(history_pane_label)
                if hasattr(HISTORY_LABEL_FONT, "getlength")
                else draw.textlength(history_pane_label, font=HISTORY_LABEL_FONT)
                if hasattr(draw, "textlength")
                else HISTORY_LABEL_FONT.getbbox(history_pane_label)[2]
                if hasattr(HISTORY_LABEL_FONT, "getbbox")
                else len(history_pane_label) * HISTORY_LABEL_FONT_SIZE
            )
            label_x = pane_x1 + (history_pane_individual_width - label_w) // 2
            label_y = (
                pane_y1 - HISTORY_LABEL_FONT_SIZE - 40
            )                                                        
            if label_y < 5:
                label_y = 5                       

                                                              
                                          
            label_padding = 2
            label_bg_color = (
                (60, 60, 60) if not USE_WHITE_BACKGROUND else (220, 220, 220)
            )
            history_draw.rectangle(
                (
                    label_x - label_padding,
                    label_y - label_padding,
                    label_x + label_w + label_padding,
                    label_y + HISTORY_LABEL_FONT_SIZE + label_padding,
                ),
                fill=label_bg_color,
            )

                             
            history_draw.text(
                (label_x, label_y),
                history_pane_label,
                font=HISTORY_LABEL_FONT,
                fill=TEXT_COLOR,
            )

            current_history_pane_x_start_coord += (
                history_pane_individual_width + HISTORY_PANE_SPACING
            )

                                                    
        if history_transition_alpha < 1.0:
                                                     
            history_img = history_img.convert("RGBA")
            alpha = int(255 * history_transition_alpha)
                                                                  
            history_alpha = Image.new("L", history_img.size, alpha)
            history_img.putalpha(history_alpha)

                                                         
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, history_img)
        img = img.convert("RGB")

    elif not SHOW_HISTORY_PANES:                                   
        mini_diff_x_start = VIDEO_WIDTH - MINI_DIFF_PANE_WIDTH + 15
        mini_diff_y_start = MAIN_PANE_Y_OFFSET                               
        current_patch_display_idx = iter_idx - 1

        for i in range(len(mini_diff_images)):
            if (
                mini_diff_y_start
                + (MINI_DIFF_HEIGHT_PER_LINE * MINI_DIFF_MAX_LINES)
                + MINI_DIFF_SPACING
                > VIDEO_HEIGHT - 30
            ):
                break

            md_img_base = mini_diff_images[i]
            is_this_one_active = i == current_patch_display_idx
            md_img_to_paste = md_img_base

            if (
                is_this_one_active
            ):                                                                      
                md_img_to_paste = draw_mini_diff(
                    raw_patch_contents_for_minidiff[i],
                    is_active=True,
                    font_mini_diff=MINI_DIFF_FONT,
                )

            img.paste(md_img_to_paste, (mini_diff_x_start, mini_diff_y_start))

            patch_label = code_states[i + 1]["patch_name"]
            if len(patch_label) > 33:
                patch_label = patch_label[:30] + "..."

            text_w_md = (
                MINI_DIFF_FONT.getlength(patch_label)
                if hasattr(MINI_DIFF_FONT, "getlength")
                else draw.textlength(patch_label, font=MINI_DIFF_FONT)
                if hasattr(draw, "textlength")
                else MINI_DIFF_FONT.getbbox(patch_label)[2]
                if hasattr(MINI_DIFF_FONT, "getbbox")
                else len(patch_label) * MINI_DIFF_TEXT_SIZE
            )
            text_x_md = (
                mini_diff_x_start + (md_img_to_paste.width // 2) - (text_w_md // 2)
            )
                          
            text_y_md = mini_diff_y_start + md_img_to_paste.height - 2
            if text_y_md + MINI_DIFF_TEXT_SIZE < VIDEO_HEIGHT - 10:
                                                               
                label_padding = 4
                label_bg_color = (
                    (60, 60, 60) if not USE_WHITE_BACKGROUND else (220, 220, 220)
                )
                draw.rectangle(
                    (
                        text_x_md - label_padding,
                        text_y_md - label_padding,
                        text_x_md + text_w_md + label_padding,
                        text_y_md + MINI_DIFF_TEXT_SIZE + label_padding,
                    ),
                    fill=label_bg_color,
                )

                                                
                draw.text(
                    (text_x_md, text_y_md),
                    patch_label,
                    font=MINI_DIFF_FONT,
                    fill=((30, 30, 30) if USE_WHITE_BACKGROUND else (220, 220, 220)),
                )
            mini_diff_y_start += (
                md_img_to_paste.height + MINI_DIFF_SPACING + MINI_DIFF_TEXT_SIZE + 5
            )

    return np.array(img)


                      
print("Creating video clip...")
animation = VideoClip(make_frame, duration=total_duration)

print(f"Writing video to {OUTPUT_VIDEO}...")
try:
    animation.write_videofile(
        OUTPUT_VIDEO, fps=FPS, codec="libx264", preset="medium", threads=4, logger="bar"
    )
except Exception as e:
    print(f"Error during video writing with libx264: {e}")
    print("Trying with mpeg4 codec as fallback...")
    try:
        animation.write_videofile(
            OUTPUT_VIDEO,
            fps=FPS,
            codec="mpeg4",
            preset="medium",
            threads=4,
            logger="bar",
        )
    except Exception as e2:
        print(f"Error during video writing with mpeg4: {e2}")
        print("Video writing failed with both codecs.")


                 
print("Cleaning up temporary directory...")
shutil.rmtree(temp_dir)
if os.path.exists(BASE_CODE_FILE) and get_file_content(BASE_CODE_FILE) == "":
    if "your_base_code_file.py" in BASE_CODE_FILE:
        print(f"Removing placeholder empty base file: {BASE_CODE_FILE}")
                                                                        

print("Done!")
