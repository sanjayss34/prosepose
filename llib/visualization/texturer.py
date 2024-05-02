# open3d renderer and camera coordinate system
# https://pytorch3d.org/docs/renderer_getting_started
from pytorch3d.renderer import TexturesVertex
import torch
import pickle as pkl

rid_to_vid = {rid: array.tolist() for rid, array in pkl.load(open('./essentials/flickrci3ds_r75_rid_to_smplx_vid.pkl', 'rb')).items()}
coarse_region_map = {
    'left hand': [3, 33, 72],
    'right hand': [23, 26, 27, 40],
    'left arm': [5, 11, 17, 34, 49, 57, 69],
    'right arm': [6, 8, 24, 32, 37, 44],
    'left foot': [1, 67],
    'right foot': [42, 65],
    'left leg': [15, 20, 21, 25, 30, 45, 53, 56, 60, 64],
    'right leg': [12, 16, 18, 19, 22, 28, 36, 46, 55, 62],
    'back': [13, 35, 38, 41, 63, 70, 2, 29, 10, 48, 14, 43],
    'head': [39, 51, 52, 66],
    'neck': [14, 43, 47, 54],
    'butt': [9, 71],
    'waist': [10, 48, 50, 68, 74],
    'waist (back)': [10, 48],
    'waist (front)': [50, 68, 74],
    'left shoulder (front)': [73],
    'left shoulder (back)': [2],
    'right shoulder (front)': [31],
    'right shoulder (back)': [29],
    'left shoulder': [2, 73],
    'right shoulder': [29, 31],
    'chest': [7, 58, 59, 61],
    'stomach': [0, 4, 68, 74]
}
hand_vertices = [v for part in ['left hand', 'right hand'] for rid in coarse_region_map[part] for v in rid_to_vid[rid]]
class Texturer():
    def __init__(
        self,
        device = 'cpu'
    ) -> None:
        super().__init__()

        self.device = device
        self.create_colors()
        self.create_num_verts()

    def create_colors(self):

        self.colors = {
            'gray': [0.7, 0.7, 0.7],
            'red': [1.0, 0.0, 0.0],
            'blue': [0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0],
            'yellow': [1.0, 1.0, 0.0],
            'paper_blue': [0.9803921568627451, 0.19607843137254902, 0.8235294117647058], #[50 / 255, 210 / 255, 250 / 255]
            'paper_red': [0.9803921568627451, 1.0, 0.36470588235294116], #[255 / 255, 93 / 255, 81 / 255]
            'pastel_blue_reversed': [142 / 255, 160 / 255, 203 / 255],
            'pastel_pink_reversed': [231 / 255, 138 / 255, 195 / 255],
            'pastel_blue': [142 / 255, 160 / 255, 203 / 255][::-1],
            'pastel_green': [102 / 255, 194 / 255, 164 / 255][::-1],
            'pastel_pink': [231 / 255, 138 / 255, 195 / 255][::-1],
            'pastel_yellow': [229 / 255, 196 / 255, 148 / 255][::-1]
        }

        for streng in range(1, 11, 1):
            sf = streng / 10
            self.colors[f'light_blue{streng}'] = [1.0, sf, sf]
            self.colors[f'light_green{streng}'] = [sf, 1.0, sf]
            self.colors[f'light_red{streng}'] = [sf, sf, 1.0]

            self.colors[f'light_yellow{streng}'] = [sf, 1.0, 1.0]
            self.colors[f'light_pink{streng}'] = [1.0, sf, 1.0]
            self.colors[f'light_turquoise{streng}'] = [1.0, 1.0, sf]

            self.colors[f'light_orange{streng}'] = [sf, 0.3+sf*0.3, 1.0]
            self.colors[f'light_aqua{streng}'] = [1.0, 0.3+sf*0.3, sf]
            self.colors[f'light_ggreen{streng}'] = [sf, 1.0, 0.3+sf*0.3]            
    
    def create_num_verts(self):
        self.num_vertices = {
            'smpl': 6890,
            'smplh': 6890,
            'smplx': 10475,
            'smplxa': 10475,
        }
    
    def get_color(self, color):
        return self.colors[color]

    def get_num_vertices(self, body_model):
        return self.num_vertices[body_model]

    def create_texture(self, vertices, color='gray', vertex_indices=None, interpolate_factor=1.0):
        """
        Create texture for a batch of vertices. 
        Vertex dimensions are expectes to be (batch_size, num_vertices, 3).
        """
        verts_rgb = torch.ones_like(vertices).to(self.device)
        if color is None:
            return TexturesVertex(verts_features=verts_rgb)
        color_rgb = self.get_color(color)
        if vertex_indices is None:
            vertex_indices = list(range(verts_rgb.shape[1]))
        if isinstance(vertex_indices[0], int):
            vertex_indices = [vertex_indices for _ in range(verts_rgb.shape[0])]
        for i in [0,1,2]:
            # print(max(vertex_indices), verts_rgb.shape)
            for j in range(verts_rgb.shape[0]):
                verts_rgb[j, vertex_indices[j], i] *= (color_rgb[i]*interpolate_factor+1*(1-interpolate_factor))
        textures = TexturesVertex(verts_features=verts_rgb)
        return textures
    
    def quick_texture(self, vertices=None, batch_size=2, body_model='smplx', colors=['blue', 'red']):
        """
        Create texture for meshes. If vertices are provided, batch size and number of vertices is taken from there.
        Otherwise, these two parameters need to be provided.
        """
        print(colors)
        if vertices is not None:
            batch_size, num_vertices, _ = vertices.shape
        else:
            num_vertices = self.get_num_vertices(body_model)
    
        dim = (batch_size, num_vertices, 3)
        verts_rgb = torch.ones(dim).to(self.device)

        for idx, color in enumerate(colors):
            # print(self.get_color(color))
            # color = 'pastel_green'
            verts_rgb_col = self.get_color(color) * num_vertices
            verts_rgb_col = torch.tensor(verts_rgb_col) \
                .reshape(-1,3).to(self.device)
            # verts_rgb[idx, hand_vertices, :] = verts_rgb_col[:len(hand_vertices),:]
            verts_rgb[idx,:,:] = verts_rgb_col
            if idx == batch_size-1:
                break

        textures = TexturesVertex(verts_features=verts_rgb)

        return textures
