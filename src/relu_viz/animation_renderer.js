// Draw exact planes from each temporally interpolated network with WebGL.
// Geometry changes only when the worker has rebuilt the actual ReLU regions.
class SurfaceRenderer {
  constructor(host, points, colors) {
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:none;touch-action:none;z-index:5';
    this.canvas.setAttribute('aria-label', 'Live training animation; drag to rotate, scroll to zoom');
    host.style.position = 'relative'; host.append(this.canvas);
    this.gl = this.canvas.getContext('webgl', {antialias: true, alpha: false});
    if (!this.gl) throw new Error('WebGL is required for smooth playback');
    const gl = this.gl;
    const vertex = `
      attribute vec2 xy;
      attribute float height, color;
      uniform float yaw, pitch, distance, aspect, zCenter, zScale, pointSize;
      varying float elevation;
      void main() {
        float z = height;
        vec3 p = vec3(xy / 3.0, (z - zCenter) * zScale);
        float cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);
        float vx = dot(p, vec3(-sy, cy, 0.0));
        float vy = dot(p, vec3(-sp*cy, -sp*sy, cp));
        float depth = distance - dot(p, vec3(cp*cy, cp*sy, sp));
        gl_Position = vec4(vx * 2.41421356 / aspect, vy * 2.41421356,
          1.002002 * depth - 0.2002002, depth);
        gl_PointSize = pointSize;
        elevation = color;
      }`;
    const fragment = `
      precision mediump float;
      varying float elevation;
      uniform vec3 palette[6];
      uniform float colorMin, colorMax;
      void main() {
        float t = clamp((elevation-colorMin)/(colorMax-colorMin), 0.0, 1.0)*5.0;
        vec3 c;
        if(t<1.0) c=mix(palette[0],palette[1],t);
        else if(t<2.0) c=mix(palette[1],palette[2],t-1.0);
        else if(t<3.0) c=mix(palette[2],palette[3],t-2.0);
        else if(t<4.0) c=mix(palette[3],palette[4],t-3.0);
        else c=mix(palette[4],palette[5],t-4.0);
        gl_FragColor=vec4(c,1.0);
      }`;
    const compile = (type, source) => {
      const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl.VERTEX_SHADER, vertex));
    gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    gl.useProgram(this.program);
    this.attributes = {};
    for (const key of ['xy','height','color']) {
      this.attributes[key] = gl.getAttribLocation(this.program,key);
    }
    this.uniforms = {};
    for (const key of ['yaw','pitch','distance','aspect','zCenter','zScale','pointSize','colorMin','colorMax','palette']) {
      this.uniforms[key] = gl.getUniformLocation(this.program,key);
    }
    gl.uniform3fv(this.uniforms.palette, colors.flatMap(hex => [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255)));
    this.points = this.buffers(points);
    this.colorMin = Math.min(...points.height); this.colorMax = Math.max(...points.height);
    this.yaw = -1.4328; this.pitch = 0.88; this.distance = 4.0;
    this.zCenter = 0.6; this.zScale = 0.65;
    this.active = false; this.drawCount = 0; this.geometryCount = 0;
    let pointer;
    this.canvas.onpointerdown = e => {pointer=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);};
    this.canvas.onpointermove = e => {
      if (!pointer) return;
      this.yaw -= (e.clientX-pointer[0])*0.008;
      this.pitch = Math.max(0.05,Math.min(1.5,this.pitch+(e.clientY-pointer[1])*0.008));
      pointer=[e.clientX,e.clientY];
    };
    this.canvas.onpointerup = this.canvas.onpointercancel = () => {pointer=null;};
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.distance=Math.max(1.5,Math.min(10,this.distance*Math.exp(e.deltaY*.001)));},{passive:false});
  }
  buffers(values) {
    const gl=this.gl, out={};
    for (const [key,value] of Object.entries(values)) {
      out[key]=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,out[key]);
      gl.bufferData(gl.ARRAY_BUFFER,value,gl.DYNAMIC_DRAW);
    }
    return out;
  }
  setFrame(mesh) {
    const count=mesh.i.length*3;
    const xy=new Float32Array(count*2),height=new Float32Array(count),color=new Float32Array(count);
    for(let triangle=0;triangle<mesh.i.length;triangle++) {
      const vertices=[mesh.i[triangle],mesh.j[triangle],mesh.k[triangle]];
      for(let corner=0;corner<3;corner++) {
        const n=triangle*3+corner,v=vertices[corner];
        xy[n*2]=mesh.x[v];xy[n*2+1]=mesh.y[v];
        height[n]=mesh.z[v];color[n]=mesh.intensity[v];
      }
    }
    const values={xy,height,color};
    if (!this.surface) this.surface=this.buffers(values);
    else for(const [key,value] of Object.entries(values)) {
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.surface[key]);
      this.gl.bufferData(this.gl.ARRAY_BUFFER,value,this.gl.DYNAMIC_DRAW);
    }
    this.count=count;this.geometryCount++;
    this.canvas.style.display='block';this.active=true;
  }
  hide() {this.active=false;this.canvas.style.display='none';}
  draw(now, orbit, showPoints) {
    if (!this.active) return;
    const gl=this.gl, ratio=Math.min(2,window.devicePixelRatio||1);
    const width=Math.round(this.canvas.clientWidth*ratio),height=Math.round(this.canvas.clientHeight*ratio);
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    if(orbit&&this.lastDraw) this.yaw+=(now-this.lastDraw)*0.00015;
    this.lastDraw=now;
    gl.viewport(0,0,width,height);gl.enable(gl.DEPTH_TEST);gl.clearColor(8/255,13/255,26/255,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    const values={yaw:this.yaw,pitch:this.pitch,distance:this.distance,aspect:width/height,zCenter:this.zCenter,zScale:this.zScale,pointSize:2*ratio,colorMin:this.colorMin,colorMax:this.colorMax};
    for(const [key,value] of Object.entries(values)) gl.uniform1f(this.uniforms[key],value);
    const bind=buffers=>{for(const [key,buffer] of Object.entries(buffers)){
      const location=this.attributes[key];gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,key==='xy'?2:1,gl.FLOAT,false,0,0);
    }};
    bind(this.surface);gl.drawArrays(gl.TRIANGLES,0,this.count);
    if(showPoints){bind(this.points);gl.drawArrays(gl.POINTS,0,this.pointCount);}
    this.drawCount++;
  }
}
