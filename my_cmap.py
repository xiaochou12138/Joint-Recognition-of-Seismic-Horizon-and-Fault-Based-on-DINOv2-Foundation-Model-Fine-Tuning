from matplotlib.colors import LinearSegmentedColormap

def get_cmap_xmw():
    text = '''private static Color[] getStrataColors(double alpha) {
        float a = (float)alpha;
        Color[] c = new Color[256];
        for (int i=0; i<256; i++) {
          if (i<8) {
            c[i] = new Color(1f,0f,0f,a);
          } else if (i<16) {
            c[i] = new Color(1f,0.5019608f,0f,a);
          } else if (i<24) {
            c[i] = new Color(1f,1f,0f,a);
          } else if (i<32) {
            c[i] = new Color(0f,1f,0f,a);
          } else if (i<40) {
            c[i] = new Color(0f,0.5019608f,0f,a);
          } else if (i<48) {
            c[i] = new Color(0f,0.2509804f,0f,a);
          } else if (i<56) {
            c[i] = new Color(0f,1f,1f,a);
          } else if (i<64) {
            c[i] = new Color(0f,0.5019608f,1f,a);
          } else if (i<72) {
            c[i] = new Color(0f,0f,1f,a);
          } else if (i<80) {
            c[i] = new Color(0f,0f,0.627451f,a);
          } else if (i<88) {
            c[i] = new Color(0f,0.5019608f,0.7529412f,a);
          } else if (i<96) {
            c[i] = new Color(1f,0.5019608f,0.5019608f,a);
          } else if (i<104) {
            c[i] = new Color(0.5019608f,0.5019608f,1f,a);
          } else if (i<112) {
            c[i] = new Color(0.5019608f,0f,1f,a);
          } else if (i<120) {
            c[i] = new Color(0.5019608f,0f,0.5019608f,a);
          } else if (i<128) {
            c[i] = new Color(1f,0.5019608f,1f,a);
          } else if (i<136) {
            c[i] = new Color(1f,0f,1f,a);
          } else if (i<144) {
            c[i] = new Color(0.5019608f,0.2509804f,0f,a);
          } else if (i<152) {
            c[i] = new Color(0.5019608f,0.5019608f,0.5019608f,a);
          } else if (i<160) {
            c[i] = new Color(0.7529412f,0.7529412f,0.7529412f,a);
          } else if (i<168) {
            c[i] = new Color(0.2509804f,0f,0.2509804f,a);
          } else if (i<176) {
            c[i] = new Color(0.90588236f,0.7294118f,0.19607843f,a);
          } else if (i<184) {
            c[i] = new Color(0.44313726f,0.58431375f,0.58431375f,a);
          } else if (i<192) {
            c[i] = new Color(0.5254902f,0.42352942f,0.4862745f,a);
          } else if (i<200) {
            c[i] = new Color(0.7176471f,0.54509807f,0.44313726f,a);
          } else if (i<208) {
            c[i] = new Color(0.5019608f,0.5019608f,0f,a);
          } else if (i<216) {
            c[i] = new Color(0.7529412f,0.7294118f,0.8784314f,a);
          } else if (i<224) {
            c[i] = new Color(0.61960787f,0.85882354f,0.9882353f,a);
          } else if (i<232) {
            c[i] = new Color(0.7372549f,0.25882354f,0.24705882f,a);
          } else if (i<240) {
            c[i] = new Color(0.8862745f,0.8509804f,0.627451f,a);
          } else if (i<248) {
            c[i] = new Color(0.60784316f,0.9411765f,0.7490196f,a);
          } else if (i<256) {
            c[i] = new Color(0.62352943f,0.79607844f,0.105882354f,a);
          }
        }
        return c;
      }'''

    lines = text.split('\n')
    colors = []
    numb_pre = 0
    for i in range(4, len(lines) - 4, 2):
        numb = lines[i].split('<')[-1].split(')')[0]
        numb = float(numb) / 256.0

        line = lines[i + 1].split('(')[-1].split(')')[0].split(',')[:3]

        rgb = []
        for j in line:
            rgb.append(float(j[:-1]))
        colors.append((rgb[0], rgb[1], rgb[2]))

    return LinearSegmentedColormap.from_list('my_cmap', colors, N=len(colors))


my_cmap = get_cmap_xmw()