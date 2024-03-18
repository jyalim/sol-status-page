#!/packages/envs/scicomp/bin/python
"""
VERSION: 0.10.1
BLAME: Jason <yalim@asu.edu>
"""
import plotly
import plotly.express as px
import pandas as pd
import numpy as np
import re
import sys

datafile = sys.argv[1]
if len(sys.argv) > 2:
  statfile = sys.argv[2]
  statdict = pd.read_csv(statfile,names='U R PD T'.split(),delim_whitespace=True).to_dict(orient='records')[0]
  stat_str = 'Queue stats: {U} Researchers, {R}/{PD}/{T} Run/Pend/Tot. Jobs'.format(**statdict)
else:
  stat_str = ''

__ANIM_MODE__ = True if len(sys.argv) > 3 else False

now = pd.Timestamp.now().strftime('%F %T')

COLS=[
  'NODELIST',
  'STATE',
  'PARTITION',
  'ALLOC_CPUS',
  'CPUS',
  'ALLOCMEM',
  'MEMORY',
  'ACTIVEMEM',
  'CPU_LOAD',
  'AVAIL_FEATURES',
  'SPECIAL',
  'gpu_info',
]

gpu_status_symbol = dict(idle='○',alloc='●')

BASE_WIDTH= 68 # HORZ PIXELS PER NODE 
BASE_DEPTH= 62 # VERT PIXELS PER NODE 
FONTSIZE  = 12

def mig_disk(mig_bounds=[0,1,2,3,4,5,6,7], alloc_mig=[], center=np.r_[0,0], radius=1, n=720):
  """
  DESCRIPTION
  -----------
  Draw a pie-graph representing Multi-Instance GPU (MIG) allocations on a GPU.

  Essentially draws a circle with either filled or empty wedges, representing
  allocated or idle MIGs.


  INPUTS
  ------
  mig_bounds, list-like of int :: defines MIG structure with ints, for instance 
  if GPU is divided into 12 equal pieces: mig_bounds=[0,1,...,12], where the 
  last value must be defined in order to correctly allocate pie-slices. Another
  example, suppose GPU is divided into 8 pieces, the first 4 twice as large
  as the last 4, then mig_bounds=[0,2,4,6,8,9,10,11,12], where
  - mig0 has bounds [0,2]
  - mig1 has bounds [2,4]
  - ...
  - mig3 has bounds [6,8]
  - mig4, the smaller class, has bounds [8,9]
  - mig5, has bounds [9,10]
  - ...
  - mig7, has bounds [11,12]
                                   

  alloc_mig, list-like of int :: defines which of the MIG as defined by
  `mig_bounds` should be marked as allocated (have their pie-wedge filled in).
  For instance, if there are 8 MIG and the first and last are allocated, then
  `alloc_mig=[0,7]`, following SLURM (and Python's) zero-indexing.
    
  center, len=2 array :: the origin of the drawn pie-graph
  radius, scalar      :: the radius of the drawn pie-graph
  n, integer          :: the number of points to draw the circle

  OUTPUTS
  -------
  pdisk, string :: a raw string of draw-instructions and coordinates that
  plotly infers to draw the pie-disk.

  psegs, string :: a raw string of draw-instructions and coordinates that 
  plotly infers to draw the ALLOCATED pie-segments.
  """
  alloc_mig  = np.array(alloc_mig)
  mig_bounds = np.array(mig_bounds)
  max_mig = mig_bounds[-1]
  # Draw main GPU circle
  OX,OY = center
  th = np.linspace(0,1,n+1,dtype=np.float64)[:-1]*2*np.pi
  x  = center + radius*np.c_[np.cos(th),np.sin(th)]
  pdisk = f"M {x[0,0]:6.3e},{x[0,1]:6.3e}" 
  for xc in x[1:]:
    pdisk += f" L{xc[0]:6.3e},{xc[1]:6.3e}"
  pdisk += f" L{x[0,0]:6.3e},{x[0,1]:6.3e} Z"
  psegs = []
  # Draw ALLOCATED MIG segments
  if len(alloc_mig) > 0:
    # determine the starting (rads0) and ending (radsN) 
    # radial bounds of all allocated MIG 
    rads0 = mig_bounds[alloc_mig]/max_mig * 2 * np.pi
    radsN = mig_bounds[alloc_mig+1]/max_mig * 2 * np.pi
    ori   = f'L{OX:6.3e},{OY:6.3e}'
    # compute cartesian coordinates from radial
    X0    = center+radius*np.c_[np.cos(rads0),np.sin(rads0)]
    XN    = center+radius*np.c_[np.cos(radsN),np.sin(radsN)]
    # compute plotly strings with sufficient resolution
    for k in range(len(X0)):
      x0,y0 = X0[k]
      xN,yN = XN[k]
      th    = np.linspace(rads0[k],radsN[k],11)[1:-1][::-1]
      xx    = center + radius*np.c_[np.cos(th),np.sin(th)]
      pseg  = f'M {x0:6.3e},{y0:6.3e} {ori} L{xN:6.3e},{yN:6.3e}'
      for x_ in xx:
        pseg += f' L{x_[0]:6.3e},{x_[1]:6.3e}'
      pseg += f' L{x0:6.3e},{y0:6.3e} Z'
      psegs.append(pseg)
  return pdisk,psegs

def get_gpu_info(rdf):
  """
  DESCRIPTION
  -----------

  From a pandas DataFrame row, build a dictionary of 
  detected GPU types and their status.
  """
  gpu_info=dict()
  try:
    for gpu_type in rdf.GRES_USED.split('gpu:')[1:]:
      name,num_alloc,_,inds = re.split('[(:)]',gpu_type)[:-1]
      num_alloc = int(num_alloc)
      is_mig = True if re.match('[12]g.[12]0gb',name) else False
      num = int(re.search(f'{name}:(\d+)',rdf.GRES).group(1))
      gpu_str = list(num*gpu_status_symbol['idle'])
      if num_alloc == num:
        gpu_str = list(num*gpu_status_symbol['alloc'])
      else:
        for ell in inds.split(','):
          if '-' in ell:
            m,n = ell.split('-')
            m,n = int(m),int(n)
            if is_mig:
              m,n = m%num,n%num
            for k in range(m,n+1):
              gpu_str[k] = gpu_status_symbol['alloc']
          elif ell == 'N/A':
            pass
          else:
            k = int(ell)
            if is_mig:
              k %= num
            gpu_str[k] = gpu_status_symbol['alloc']
      gpu_info[name] = dict(
        name      = name,
        num       = num,
        num_idle  = num-num_alloc,
        num_alloc = num_alloc,
        alloc_ind = inds,
        gpu_str   = ''.join(gpu_str),
        is_mig    = is_mig
      )
  except Exception as ex:
    print('GPU string parsing exception: ', ex, rdf.GRES_USED)
  return gpu_info
  
def get_gpu_summary(rdf):
  gpu_summary=''
  try:
    if rdf.gpu_info != {}:
      gpu_summary='<br> <b>GPUs</b>: '
      for k,gpu in enumerate(rdf.gpu_info):
        name,alloc,total = [ rdf.gpu_info[gpu][k] for k in 'name num_alloc num'.split() ]
        if k > 0:
          gpu_summary += ', '
        gpu_summary += f'{name}:{alloc}/{total}'
  except Exception as ex:
    print('GPU Info parsing exception: ', ex, rdf.gpu_info)
  return gpu_summary

HOVERTEMPLATE = r"""<extra></extra>
<b>%{customdata[0]}</b>            
<br>
<b>State</b>: %{customdata[1]}    
<br>
<b>Partition</b>: %{customdata[2]}
<br>
<b>Alloc. / Total Cores</b>: %{customdata[3]}/%{customdata[4]}
<br>
<b>Alloc. / Total RAM</b>: %{customdata[5]:,d}/%{customdata[6]:,d} (GiB) 
<br>
<b>Active RAM</b>: %{customdata[7]:,d} (GiB) 
<br>
<b>CPU Load</b>: %{customdata[8]}
<br>
<b>Features</b>: %{customdata[9]}  
%{customdata[10]}
"""
 
AGG_DICT = { k : 'first' for k in COLS }
AGG_DICT['PARTITION'] = ','.join
AGG_DICT.pop('NODELIST',None)

STATE_SUFFIXES='*~#!%$@^-+'

BASE_STATE_COLOR_MAP = {
  'idle'             : '#00ff00',
  'mixed'            : '#ff8800',
  'allocated'        : '#ff0000',
  'completing'       : '#ffff00',
  'down'             : '#777777',
  'drained'          : '#00aacc',
  'draining'         : '#00ddff',
  'inval'            : '#999900',
  'maint'            : '#ff5f15',
  'planned'          : '#ffb300',
  'reboot'           : '#ffffff',
  'reboot_issued'    : '#ffffff',
  'reboot_requested' : '#ffffff',
  'power_down'       : '#ffffff',
  'powered_down'     : '#ffffff',
  'powering_down'    : '#ffffff',
  'powering_up'      : '#ffffff',
  'reserved'         : '#990000',
  'unknown'          : '#999999',
  'fail'             : '#999999',
  'failing'          : '#999999',
  'future'           : '#eeeeee',
}

BASE_STATE_TEXTFONT_COLOR_MAP = {
  'idle'             : '#000000',
  'mixed'            : '#000000',
  'allocated'        : '#ffffff',
  'completing'       : '#000000',
  'down'             : '#ffffff',
  'drained'          : '#000000',
  'draining'         : '#000000',
  'inval'            : '#ff0000',
  'maint'            : '#000000',
  'planned'          : '#000000',
  'reboot'           : '#000000',
  'reboot_issued'    : '#000000',
  'reboot_requested' : '#000000',
  'power_down'       : '#000000',
  'powered_down'     : '#000000',
  'powering_down'    : '#000000',
  'powering_up'      : '#000000',
  'reserved'         : '#ffffff',
  'unknown'          : '#ffff00',
  'fail'             : '#ffff00',
  'failing'          : '#ffff00',
  'future'           : '#00ff00',
}

STATE_COLOR_MAP=dict()
for state,value in BASE_STATE_COLOR_MAP.items():
  STATE_COLOR_MAP[f'{state}'] = value
  for suffix in STATE_SUFFIXES:
    STATE_COLOR_MAP[f'{state}{suffix}'] = value

STATE_TEXTFONT_COLOR_MAP=dict()
for state,value in BASE_STATE_TEXTFONT_COLOR_MAP.items():
  STATE_TEXTFONT_COLOR_MAP[f'{state}'] = value
  for suffix in STATE_SUFFIXES:
    STATE_TEXTFONT_COLOR_MAP[f'{state}{suffix}'] = value

SCATTER_OPTS = dict(
  x='x',
  y='y',
  color='STATE',
  custom_data=COLS,
  color_discrete_map=STATE_COLOR_MAP,
  text='NODELIST',
  category_orders={'STATE':STATE_COLOR_MAP.keys()},
)

TITLE_DICT = dict(
  text=f'<b>Sol Supercomputer Node Status</b> {now} <br>{stat_str}',
  y=0.975,
  x=0.01,
  xanchor='left',
  yanchor='top',
)

if __ANIM_MODE__:
  TITLE_DICT['text'] = f'<b>Sol Supercomputer Node Status</b> {datafile.split("/")[-1].split(".")[0].split("node-status-")[-1]} <br>{stat_str}'
  

TRACE_OPTS = dict(
  hovertemplate=HOVERTEMPLATE,
  marker_line_width = 2,  
  marker_size       = 54,  
  marker_symbol     = 'square',  
  textposition      = 'middle center',  
  line_color        = '#000000',
  textfont=dict(family=['monospace'],size=FONTSIZE,color='#000000'),
)

df = pd.read_csv(datafile,delim_whitespace=True,escapechar='\\')
df['SPECIAL']    = ''
df['CPU_LOAD']   = df['CPU_LOAD'].fillna(0)
df['ALLOCMEM']   = df['ALLOCMEM'].fillna(0)/1024
df['FREE_MEM']   = df['FREE_MEM'].fillna(0)/1024
df['MEMORY']     = df['MEMORY']/1024
df['ALLOC_CPUS'] = df.apply(lambda x: int(x['CPUS(A/I/O/T)'].split('/')[0]), axis=1).fillna(0)
df['ACTIVEMEM' ] = df['MEMORY'] - df['FREE_MEM']

df.loc[df['GRES']=='(null)','GRES_USED'] = ''
df.loc[df['GRES']!='(null)','GRES'] = df.loc[df['GRES']!='(null)','GRES'].str.replace(r'(S:0-1)','',regex=False)

df.loc[df['TIMESTAMP'] != 'Unknown','TIMESTAMP'] = '('+df.loc[df['TIMESTAMP']!='Unknown','TIMESTAMP']+') '
df.loc[df['REASON'] != 'none','SPECIAL'] += '<br> <b>Reason</b>: '+df.loc[df['REASON'] != 'none','TIMESTAMP']+df.loc[df['REASON'] != 'none','REASON']

df['gpu_info'] = df.apply(lambda x: get_gpu_info(x),axis=1)
df['SPECIAL']  = df.apply(lambda x: get_gpu_summary(x),axis=1)

df = df.groupby('NODELIST').agg(AGG_DICT).reset_index()
L  = len(df)
N  = int(np.sqrt(L)+1)
M  = int(L/N+1)
df['x'] = df.apply(lambda x: x.name//M,axis=1)
df['y'] = df.apply(lambda x: x.name%M, axis=1)

gdf = df.loc[df['gpu_info']!={},['x','y','gpu_info','STATE']]

fig = px.scatter(df,**SCATTER_OPTS)
fig.layout.xaxis['visible'] = False
fig.layout.yaxis['visible'] = False
fig.layout.paper_bgcolor = '#eeeeee'
fig.layout.plot_bgcolor  = '#eeeeee'
fig.layout.width  = BASE_WIDTH * N # /11) if M > 11 else BASE_WIDTH
fig.layout.height = BASE_DEPTH * M # /11) if N > 11 else BASE_DEPTH
print('len M N w h')
print(len(df), M, N, fig.layout.width, fig.layout.height)
fig.update_traces(**TRACE_OPTS)
pshapes = []
for i,r in gdf.iterrows():
  c = STATE_TEXTFONT_COLOR_MAP[r.STATE]
  # TODO: make more flexible 
  if len(r.gpu_info) > 1:
    gpus_alloc = [[],[],[],[]]
    g2sm20g = r.gpu_info['2g.20gb']
    g1sm20g = r.gpu_info['1g.20gb']
    for k,status in enumerate(g2sm20g['gpu_str']):
      if status == gpu_status_symbol['alloc']:
        gpu_index      = k//3
        gpu_sub_index  = k%3
        gpus_alloc[gpu_index].append(gpu_sub_index)
    for k,status in enumerate(g1sm20g['gpu_str']):
      if status == gpu_status_symbol['alloc']:
        gpu_index      = k
        gpu_sub_index  = 3
        gpus_alloc[k].append(gpu_sub_index)
    for k in range(len(gpus_alloc)):
      pdisk,psegs = mig_disk(center=np.r_[r.x-0.32+k*0.21,r.y-0.32], radius=0.095, alloc_mig=gpus_alloc[k], mig_bounds=[0,2,4,6,7], n=101)
      pshapes.append(dict(type="path",path=pdisk,line_color=c,line=dict(width=1.2)))
      for pseg in psegs:
        pshapes.append(
          dict(type="path",path=pseg,fillcolor=c,line_color=c,line=dict(width=0))
        )
  else:
    name = list(r.gpu_info.keys())[0]
    gpu_info= r.gpu_info[name]
    gpu_str = gpu_info['gpu_str'] 
    if len(gpu_str) == 8:
      fig.add_annotation(
        x=r.x-0.45,
        y=r.y-0.04,
        text=gpu_str[:4],
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(size=FONTSIZE,color=c),
      )
      gpu_str = gpu_str[4:]

    fig.add_annotation(
      x=r.x-0.45,
      y=r.y-0.16,
      text=gpu_str,
      showarrow=False,
      xanchor='left',
      yanchor='top',
      font=dict(size=FONTSIZE,color=c),
    )
fig.update_layout(
  autosize=True,
# width=500,
# height=500,
  margin=dict(
      l=0,
      r=0,
      b=0,
      t=50,
#     pad=4
  ),
# paper_bgcolor="white",
  xaxis_range= [-0.5,df['x'].values.max()+0.5],
  yaxis_range= [-0.5,df['y'].values.max()+0.5],
  title=TITLE_DICT,
  shapes=pshapes 
)
for scatter in fig.data:
  legendgroup = scatter['legendgroup']
  if legendgroup in STATE_TEXTFONT_COLOR_MAP:
    scatter['textfont']['color'] = STATE_TEXTFONT_COLOR_MAP[legendgroup]
  else:
    if not __ANIM_MODE__:
      with open('crontab.diag.log','a') as fh:
        print(f'[{now}] undefined legendgroup: {legendgroup}', file=fh)
  
if not __ANIM_MODE__: 
  plotly.offline.plot(fig,filename="sol2.html",include_plotlyjs="cdn")
else:
  fig.write_image(f"anim/{datafile.split('/')[-1].split('.')[0]}.png",scale=2)
