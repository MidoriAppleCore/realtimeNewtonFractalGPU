import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import re
import pygame
import sympy as s
from sympy import symbols, I, sympify, expand, diff, sin, cosh, cos, sinh, Symbol
from sympy import im as sim
from sympy import re as sre


class NewtonGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.center_x = 0  # Default center x-coordinate
        self.center_y = 0  # Default center y-coordinate
        self.zoom = 1  # Default zoom level

        self._center_x = 0  # Default center x-coordinate
        self._center_y = 0  # Default center y-coordinate
        self._zoom = 1  # Default zoom level

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        # placeholder
        self.program = None

        # Buffers
        self.color_data = np.empty((self.height, self.width, 4), dtype=np.uint8)  # RGBA color data
        self.color_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.color_data.nbytes)

    def loadKernel(self, kernelProgram):
        self.program = kernelProgram

    def getImage(self, center_x, center_y, zoom, zoom_factor, param_values):
        param_array = np.array(param_values, dtype=np.float32)
        param_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=param_array.nbytes)
        cl.enqueue_copy(self.queue, param_buffer, param_array)
        assert param_buffer.size == param_array.nbytes

        self.program.newton_fractal(
            self.queue, 
            self.color_data.shape[:-1],  # Only width and height
            None, 
            self.color_buffer, 
            np.uint32(self.width), 
            np.uint32(self.height), 
            param_buffer,  # Pass the parameter values to the kernel
            np.float32(center_x), 
            np.float32(center_y), 
            np.float32(zoom)
        )
        cl.enqueue_copy(self.queue, self.color_data, self.color_buffer).wait()

    def getContext(self):
        return self.context

    @property
    def center(self):
        return self._center_x, self._center_y

    @center.setter
    def center(self, value):
        self._center_x, self._center_y = value

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = value

default_kernel = """
// Direct access to w[] inside the function
float re(int idx, __global const float w[]) {
    return w[2 * idx];
}

float im(int idx, __global const float w[]) {
    return w[2 * idx + 1];
}
__kernel void newton_fractal(
    __global uchar4 *colors,
    const unsigned int width,
    const unsigned int height,
    __global const float w[],  // w is now an interleaved array of real and imaginary values
    const float fractal_center_x,
    const float fractal_center_y,
    const float fractal_zoom
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = y * width + x;

    float aspect_ratio = (float)width / (float)height;
    float scale_factor = 3.5f / aspect_ratio / fractal_zoom;
    float shift_x = fractal_center_x - 1.75f / aspect_ratio / fractal_zoom;
    float shift_y = fractal_center_y + 1.75f / fractal_zoom;

    float zx = shift_x + (float)x / (float)width * scale_factor;
    float zy = shift_y - (float)y / (float)height * scale_factor;

    ushort iteration = 0;
    const float tolerance = 1.0e-4;

    while (iteration < 60) {
        [put it here]

        // Newton's update: z = z - f(z) / f'(z)
        float denom = f_prime_r_x * f_prime_r_x + f_prime_r_y * f_prime_r_y + f_prime_i_x * f_prime_i_x + f_prime_i_y * f_prime_i_y;
        float zx_new = zx - (f_r * f_prime_r_x + f_i * f_prime_i_x) / denom;
        float zy_new = zy - (f_i * f_prime_r_x - f_r * f_prime_i_x) / denom;

        if ((zx - zx_new) * (zx - zx_new) + (zy - zy_new) * (zy - zy_new) < tolerance * tolerance) break;

        zx = zx_new;
        zy = zy_new;

        iteration++;
    }

    //uchar red = (uchar)(cos((zy * 2 - iteration/5+ 31)*.18) * 205.0f + 55.0f);
    //uchar green = (uchar)(sin((zx * 2 - iteration/5 + 53)*.13) * 100.0f + 55.0f);
    //uchar blue = (uchar)(cos((zy * zx * 5 - iteration/5 + 50)*.20) * 255.0f);
    
    // Example of using magnitude and phase for coloring
    float magnitude = sqrt(zx*zx + zy*zy);
    float phase = atan2(zy, zx);

    uchar red = (uchar)(sin(magnitude) * 255.0f);
    uchar blue = (uchar)(cos(phase / M_PI) * 128.0f + 128.0f);
    uchar green = (uchar)(sin(255.0f - magnitude) * 255.0f);

    red = red - (uchar)iteration/10;
    blue = blue - (uchar)iteration/10;
    green = green - (uchar)iteration/10;

    colors[index] = (uchar4)(red, green, blue, 255);
}
"""
func_kernel = """
// Direct access to w[] inside the function
float re(int idx, __global const float w[]) {
    return w[2 * idx];
}

float im(int idx, __global const float w[]) {
    return w[2 * idx + 1];
}
__kernel void newton_fractal(
    __global uchar4 *colors,
    const unsigned int width,
    const unsigned int height,
    __global const float w[],  // w is now an interleaved array of real and imaginary values
    const float fractal_center_x,
    const float fractal_center_y,
    const float fractal_zoom
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = y * width + x;

    float aspect_ratio = (float)width / (float)height;
    float scale_factor = 3.5f / aspect_ratio / fractal_zoom;
    float shift_x = fractal_center_x - 1.75f / aspect_ratio / fractal_zoom;
    float shift_y = fractal_center_y + 1.75f / fractal_zoom;

    float zx = shift_x + (float)x / (float)width * scale_factor;
    float zy = shift_y - (float)y / (float)height * scale_factor;

    // Your C code generated by getHarmonicsCode is inserted here
    [put it here]

    // Example of using magnitude and phase for coloring
    float magnitude = sqrt(f_r*f_r + f_i*f_i);
    float phase = atan2(f_i, f_r);

    uchar red = (uchar)(magnitude * 255.0f);
    uchar green = (uchar)(phase / M_PI * 128.0f + 128.0f);
    uchar blue = (uchar)(255.0f - magnitude * 255.0f);
    // Generate colors based on the output of the function
    //uchar red = (uchar)(cos((zy * 2 - f_i + 31)*.18) * 205.0f + 55.0f);
    //uchar green = (uchar)(sin((zx * 2 - f_r + 53)*.13) * 100.0f + 55.0f);
    //uchar blue = (uchar)(cos((zy * zx * 5 - f_i * f_r + 50)*.20) * 255.0f);

    colors[index] = (uchar4)(red, green, blue, 255);
}
"""
class Kernel:
    def __init__(self, f="z*z*z-1", kernel_code=default_kernel, simplifyHarmonics=False):
        self.f = f
        self.kernel_code = kernel_code
        #self.kernel_code = func_kernel
        self.simplifyHarmonics = simplifyHarmonics

    #Take in a expression string and return the harmonic functions as C code that can be injected into an openCL kernel

    def getHarmonicsCode(self, expr_str, var='z', real_var='x', imag_var='y', params=['w0', 'w1', 'w2']):
        import sympy as sp
        import re  # Make sure to import Python's re module

        x, y = sp.symbols(f'{real_var} {imag_var}', real=True)
        w_symbols = sp.symbols(' '.join(params))

        substituted_expr = sp.sympify(expr_str).subs(var, x + sp.I*y)

        # Extract real and imaginary parts
        f_real, f_imag = substituted_expr.as_real_imag()

        # Expand the results
        f_real = f_real.expand()
        f_imag = f_imag.expand()

        print("f real f imag")
        print(f_real, f_imag)

        f_prime_real_x = sp.diff(f_real, x)
        f_prime_real_y = sp.diff(f_real, y)
        f_prime_imag_x = sp.diff(f_imag, x)
        f_prime_imag_y = sp.diff(f_imag, y)

        if self.simplifyHarmonics == True:
            print("simplify real x")
            f_prime_real_x =  f_prime_real_x.simplify()
            print("simplify real y")
            f_prime_real_y =  f_prime_real_y.simplify()
            print("simplify imag x")
            f_prime_imag_x =  f_prime_imag_x.simplify()
            print("simplify imag y")
            f_prime_imag_y =  f_prime_imag_y.simplify()

        c_code_dict = {
            'f_r': sp.ccode(f_real).replace('x', 'zx').replace('y', 'zy'),
            'f_i': sp.ccode(f_imag).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_r_x': sp.ccode(f_prime_real_x).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_r_y': sp.ccode(f_prime_real_y).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_i_x': sp.ccode(f_prime_imag_x).replace('x', 'zx').replace('y', 'zy'),
            'f_prime_i_y': sp.ccode(f_prime_imag_y).replace('x', 'zx').replace('y', 'zy')
        }

        for key in c_code_dict:
            c_code_dict[key] = re.sub(r'pow\(([^,]+),\s*([^)]+)\)', r'pow((float)\1, (float)\2)', c_code_dict[key])
            c_code_dict[key] = c_code_dict[key].replace(' 0,', ' (float)0,').replace(' 0)', ' (float)0)')
            c_code_dict[key] = re.sub(r're\(w(\d+)\)', r're(\1, w)', c_code_dict[key])
            c_code_dict[key] = re.sub(r'im\(w(\d+)\)', r'im(\1, w)', c_code_dict[key])

        c_code_str = f"float f_r = (float)({c_code_dict['f_r']});\n"
        c_code_str += f"float f_i = (float)({c_code_dict['f_i']});\n"
        c_code_str += f"float f_prime_r_x = (float)({c_code_dict['f_prime_r_x']});\n"
        c_code_str += f"float f_prime_r_y = (float)({c_code_dict['f_prime_r_y']});\n"
        c_code_str += f"float f_prime_i_x = (float)({c_code_dict['f_prime_i_x']});\n"
        c_code_str += f"float f_prime_i_y = (float)({c_code_dict['f_prime_i_y']});\n"

        print(c_code_str)
        return c_code_str

    def generateKernel(self, context):
        self.program = cl.Program(context(), self.kernel_code.replace('[put it here]', self.getHarmonicsCode(self.f)))
        self.program.build()
        return self.program

    def saveKernel(self, filepath):
        pass


class PygameRender:
    def __init__(self, width=800, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.HWSURFACE)
        pygame.display.set_caption('Newton Fractal Viewer')
        self.clock = pygame.time.Clock()

    def blit(self, colors):
        # Create the initial surface from the raw color data
        image_data = np.array(colors).astype(np.uint8).reshape(colors.shape[0], colors.shape[1], 4)
        initial_surface = pygame.surfarray.make_surface(image_data[:, :, :3])
        
        # Check if the fractal dimensions match the screen dimensions
        if initial_surface.get_width() != self.width or initial_surface.get_height() != self.height:
            # If they don't match, rescale the surface
            rescaled_surface = pygame.transform.smoothscale(initial_surface, (self.width, self.height))
        else:
            rescaled_surface = initial_surface
        
        # Blit the resulting surface to the screen
        self.screen.blit(rescaled_surface, (0, 0))
        pygame.display.flip()

class PygameInput:
    def __init__(self, newton_generator, pygame_render, pan_speed=0.05, zoom_factor=1.1):
        self.newton_gen = newton_generator
        self.render = pygame_render
        self.pan_speed = pan_speed
        self.zoom_factor = zoom_factor
        self.record_mode = False
        self.output_folder = None
        self.counter = 0

    def handle_events(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Check for the 'r' key press
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.record_mode = not self.record_mode
                if self.record_mode:
                    # Create a new folder based on the current date
                    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.output_folder = f"outputs_{date_str}"
                    os.makedirs(self.output_folder, exist_ok=True)
                    self.counter = 0  # reset counter

        # Panning and Zooming
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.newton_gen.center = (self.newton_gen.center[0], self.newton_gen.center[1] - self.pan_speed / self.newton_gen.zoom)
        if keys[pygame.K_LEFT]:
            self.newton_gen.center = (self.newton_gen.center[0], self.newton_gen.center[1] + self.pan_speed / self.newton_gen.zoom)
        if keys[pygame.K_UP]:
            self.newton_gen.center = (self.newton_gen.center[0] - self.pan_speed / self.newton_gen.zoom, self.newton_gen.center[1])
        if keys[pygame.K_DOWN]:
            self.newton_gen.center = (self.newton_gen.center[0] + self.pan_speed / self.newton_gen.zoom, self.newton_gen.center[1])
        if keys[pygame.K_MINUS]:
            self.newton_gen.zoom /= self.zoom_factor
        if keys[pygame.K_EQUALS]:
            self.newton_gen.zoom *= self.zoom_factor

        return running


if __name__=="__main__":
    SCREEN_WIDTH = int(1000/1)  # Example screen width
    SCREEN_HEIGHT = int(1000/1)  # Example screen height

    FRACTAL_WIDTH = int(800/1)  # Example fractal width for subsampling
    FRACTAL_HEIGHT = int(800/1)  # Example fractal height for subsampling

    param_values = [1.0, 0.000, 1.000, 0, 1.0, 0]

    #z3 = Kernel(f="cos(I*sin(w0*z*z*z-1))*sin(re(w0)) + z*sin(z*z*z*w1)+sin(z*w0) - w2*z*I")

    #takes a while
    #z3 = Kernel(f="(cos(sin(z))+(z*z*z*w0-1)*w1*I)*(1/sin(z))")

    #z3 = Kernel(f="(z*z*z-1) + (z*z*z*z-1)")

    z3 = Kernel(f="cos(I*sin(I*z))*z+(w1+w2+cos(z))")

    #does not work
    #z3 = Kernel(f="z*cos(sin(z*w1)*w1)-1")
    #z3 = Kernel(f="(z*z*z*w0-1)*(1+sin(w1)) + (sin(z*w0)/sin(z*w)-1)*(1+sin(w1))")
    #z3 = Kernel(f="(z*z*z*w1-w2)*(1-sin(w1))*(z*z*z-w0)*sin(w1)")
    #z3 = Kernel(f="(cos(z)*w1)*z*z*z*w2-1")

    new = NewtonGenerator(FRACTAL_WIDTH, FRACTAL_HEIGHT)
    new.loadKernel(z3.generateKernel(new.getContext))

    render = PygameRender(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    input_handler = PygameInput(new, render)

    running = True

    c = 0
    while running:
        running = input_handler.handle_events()
        new.getImage(new.center[0], new.center[1], new.zoom, 1, param_values=param_values)
        render.blit(new.color_data)
        c = c + .01

        param_values[0] = sin(c/20)
        param_values[2] = sin(c/200)
        param_values[1] = c/200
        param_values[3] = c/200
        param_values[4] = cos(c)+c/100
        param_values[4] = sin(c)+c/100
