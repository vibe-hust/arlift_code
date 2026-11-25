"""Render a schematic README animation with Pillow (no model execution).

Run: python docs/render_method_intro.py
Requires: Python >= 3.10 and Pillow >= 10. Fonts fall back across macOS/Linux.
The animation illustrates control flow, not measured speed or quality.
"""
from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

W, H, SCALE = 1200, 720, 2
BG = '#FAFBFD'
INK = '#17243B'
MUTED = '#69758A'
LINE = '#DCE3ED'
WHITE = '#FFFFFF'
PALE = '#F0F4FA'
ORANGE = '#C97513'
FONTS = {}


def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in FONTS:
        names = (['/System/Library/Fonts/Menlo.ttc', 'DejaVuSansMono.ttf'] if mono else
                 ['/System/Library/Fonts/Supplemental/Arial Bold.ttf', 'DejaVuSans-Bold.ttf'] if bold else
                 ['/System/Library/Fonts/Supplemental/Arial.ttf', 'DejaVuSans.ttf'])
        for name in names:
            try:
                FONTS[key] = ImageFont.truetype(name, size * SCALE)
                break
            except OSError:
                continue
        else:
            raise RuntimeError('Install DejaVu Sans or edit the font paths in this script.')
    return FONTS[key]


def mix(a, b, t):
    def rgb(c): return tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
    return tuple(round(x + (y-x)*t) for x, y in zip(rgb(a), rgb(b)))


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W*SCALE, H*SCALE), BG)
        self.d = ImageDraw.Draw(self.im)

    def rect(self, box, fill=WHITE, outline=None, width=1, radius=16):
        self.d.rounded_rectangle(tuple(round(v*SCALE) for v in box),
                                 radius=radius*SCALE, fill=fill, outline=outline,
                                 width=width*SCALE)

    def line(self, points, fill=LINE, width=2):
        self.d.line([(round(x*SCALE), round(y*SCALE)) for x, y in points],
                    fill=fill, width=width*SCALE, joint='curve')

    def circle(self, x, y, r, fill, outline=None):
        self.d.ellipse(((x-r)*SCALE, (y-r)*SCALE, (x+r)*SCALE, (y+r)*SCALE),
                       fill=fill, outline=outline, width=2*SCALE)

    def text(self, x, y, value, size=20, color=INK, bold=False, mono=False, center=False):
        f = font(size, bold, mono)
        box = self.d.textbbox((0, 0), value, font=f)
        width = box[2] - box[0]
        px = x*SCALE - (width/2 if center else 0)
        assert px >= -1 and px+width <= W*SCALE+1, (value, px, width)
        self.d.text((px, y*SCALE-box[1]), value, fill=color, font=f)

    def arrow(self, pts, color=LINE, width=3, travel=None):
        self.line(pts, color, width)
        (x0,y0),(x,y)=pts[-2:]
        a=math.atan2(y-y0,x-x0)
        ends=[(x-9*math.cos(a+d), y-9*math.sin(a+d)) for d in [-0.5,0.5]]
        self.line([ends[0],(x,y),ends[1]],color,width)
        if travel is not None:
            lengths=[math.hypot(q[0]-p[0],q[1]-p[1]) for p,q in zip(pts,pts[1:])]
            distance=(travel%1)*sum(lengths)
            for p,q,n in zip(pts,pts[1:],lengths):
                if distance <= n:
                    k=distance/n if n else 0
                    self.circle(p[0]+k*(q[0]-p[0]),p[1]+k*(q[1]-p[1]),6,color)
                    break
                distance-=n

    def chip(self,x,y,w,label,color=ACCENT if 'ACCENT' in globals() else '#2563EB',active=True):
        self.rect((x,y,x+w,y+32),mix(color,WHITE,0.92) if active else PALE,None,radius=8)
        self.text(x+w/2,y+9,label,14,color if active else MUTED,bold=True,center=True)

    def card(self,x,y,w,h,label,active=True):
        self.rect((x,y+3,x+w,y+h+3),'#EBEFF5',None,radius=18)
        self.rect((x,y,x+w,y+h),WHITE,ACCENT if active else LINE,2 if active else 1,18)
        self.text(x+20,y+22,label,19,bold=True)

    def finish(self):
        return self.im.resize((W,H),Image.Resampling.LANCZOS)


def frame(stage,u,elapsed,poster=False):
    c=Canvas()
    c.text(55,30,PROJECT+'  /  METHOD IN MOTION',15,ACCENT,bold=True)
    c.chip(945,23,200,'FROZEN MODEL WEIGHTS',ACCENT)
    c.text(55,73,TITLE,36,bold=True)
    c.text(55,125,SUBTITLE,21,MUTED)
    for i,label in enumerate(STEPS):
        x=55+i*280
        active=i==stage
        c.rect((x,177,x+250,225),mix(ACCENT,WHITE,0.92) if active else WHITE,
               ACCENT if active else LINE,2 if active else 1,12)
        c.circle(x+25,201,12,ACCENT if active else '#E7ECF4')
        c.text(x+25,194,str(i+1),14,WHITE if active else MUTED,bold=True,center=True)
        c.text(x+49,193,label,18,ACCENT if active else MUTED,bold=True)
    draw_method(c,stage,u,poster)
    c.rect((55,597,1145,678),mix(ACCENT,WHITE,0.94),None,radius=14)
    c.text(75,614,CAPTIONS[stage],21,bold=True)
    c.text(75,649,FOOTNOTE,16,MUTED)
    c.rect((55,700,1145,704),'#E7ECF4',None,radius=2)
    c.rect((55,700,55+max(4,1090*elapsed),704),ACCENT,None,radius=2)
    return c.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'assets')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frames=[]
    for n in range(160):
        stage=n//40
        u=(n%40)/39
        frames.append(frame(stage,u,n/159))
    samples=Image.new('RGB',(W,H*4))
    for s in range(4): samples.paste(frames[s*40+25],(0,s*H))
    palette=samples.quantize(colors=192,method=Image.Quantize.MEDIANCUT)
    indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    path=args.output_dir/'method-intro.gif'
    indexed[0].save(path,save_all=True,append_images=indexed[1:],duration=100,loop=0,
                    optimize=True,disposal=1)
    frame(3,1,1,poster=True).save(args.output_dir/'method-overview.png',optimize=True)
    print(f'{PROJECT}: {path} ({path.stat().st_size/1024:.0f} KiB), 1200x720, 16 seconds')


PROJECT='ARLIFT'
ACCENT='#7950C7'
TITLE='Keep the structure. Generate at higher resolution.'
SUBTITLE='A low-resolution draft guides a high-resolution denoising branch.'
STEPS=['Generate a draft','Lift the reference','Guide denoising','Decode the result']
CAPTIONS=[
 'Generate a low-resolution draft to establish scene structure and motion.',
 'Upsample the clean draft latent to match the high-resolution prediction.',
 'Add a blurred structural correction to the high-resolution clean prediction.',
 'Decode the guided result at 2× width and 2× height, with frozen model weights.',
]
FOOTNOTE='Krea / LongLive path shown. Latent grids are schematic illustrations, not generated videos.'


def latent(c,x,y,w,h,nx,ny,clarity=1,detail=False,seed=0):
    colors=['#EBE5F7','#D7C6EF','#B8A0DD','#9574C2','#7950C7','#5A398F']
    for j in range(ny):
        for i in range(nx):
            xx=(i+0.5)/nx; yy=(j+0.5)/ny
            shape=0.34+0.22*math.sin(xx*5.8+yy*2.3)+0.27*math.exp(-((xx-0.64)**2+(yy-0.48)**2)*22)
            if detail: shape+=0.07*math.sin(xx*42)*math.cos(yy*38)
            noise=((i*37+j*19+i*j*7+seed*11)%101)/100
            val=shape*clarity+noise*(1-clarity)
            idx=max(0,min(5,int(val*6)))
            c.rect((x+i*w/nx,y+j*h/ny,x+(i+1)*w/nx-0.6,y+(j+1)*h/ny-0.6),colors[idx],None,radius=0)


def draw_method(c,s,u,poster):
    c.card(55,270,250,248,'Low-res draft',s==0)
    c.text(180,316,'Clean latent reference',15,MUTED,center=True)
    latent(c,108,354,144,96,12,8,clarity=min(1,0.25+u*1.25) if s==0 else 1)
    c.text(180,479,'Structure + motion',16,ACCENT,bold=True,center=True)

    c.card(395,270,410,248,'High-res branch',s in [1,2])
    c.text(600,322,'Independent noise' if s<2 else 'Guided clean prediction',15,MUTED,center=True)
    if s==0: clarity=0
    elif s==1: clarity=0.15
    elif s==2: clarity=0.15+0.85*u
    else: clarity=1
    latent(c,480,345,240,160,24,16,clarity=clarity,detail=s>=2,seed=3)
    c.arrow([(305,398),(395,398)],ACCENT if s in [1,2] or poster else LINE,
            travel=u*2 if s in [1,2] else None)
    c.text(350,364,'resize',15,ACCENT if s in [1,2] else MUTED,center=True)
    if s==2:
        scan=345+u*160
        c.line([(475,scan),(725,scan)],'#FAFCFF',3)

    c.card(895,270,250,248,'Refined output',s==3)
    c.text(1020,316,'2× width · 2× height',16,ACCENT,bold=True,center=True)
    latent(c,922,350,196,131,24,16,clarity=1 if s==3 or poster else 0.2,detail=True,seed=3)
    c.arrow([(805,398),(895,398)],ACCENT if s==3 or poster else LINE,
            travel=u*2 if s==3 else None)
    c.text(850,364,'decode',15,ACCENT if s==3 else MUTED,center=True)

    c.text(600,546,'clean prediction  +  λ × blur(upsampled draft − clean prediction)',21,
           ACCENT if s==2 or poster else MUTED,bold=True,center=True)
    c.text(600,577,'Spatial guidance preserves the draft structure while the high-res branch generates detail.',16,MUTED,center=True)


if __name__=='__main__': main()
