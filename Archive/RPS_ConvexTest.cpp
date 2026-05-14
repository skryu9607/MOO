/*
 * RPS Convex Verification Test (Integrated with HTML Visualization)
 * ==================================================================
 * Runs RPS on a guaranteed-convex 3-objective problem, verifies
 * non-dominance, then emits a single self-contained HTML file with:
 *   1. Weight-simplex sampling plot
 *   2. Three pairwise 2-D Pareto-front projections
 *   3. Interactive 3-D Pareto-front view (mouse-drag rotation)
 *   4. Non-dominance verification table
 *
 * PROBLEM:
 *   min  f_i(x) = ||x - a_i||^2,  i = 1,2,3,  x in R^2
 *   a_1=(0,0), a_2=(10,0), a_3=(5, 5*sqrt(3))   (equilateral, side=10)
 *
 *   Closed-form:  x*(w) = sum w_i a_i
 *                 f_i    = ||x*(w) - a_i||^2
 *
 * Dependencies: Gurobi C++, Eigen3
 *
 * Compile:
 g++ -m64 -O2 RPS_ConvexTest.cpp -o RPS_ConvexTest_copy \
       -I/opt/gurobi1300/linux64/include \
        -L/opt/gurobi1300/linux64/lib \
        -I/usr/include/eigen3 \
        -lgurobi_c++ -lgurobi130 -lpthread
 *
 * Run:
 *   ./RPS_ConvexTest
 *   -> terminal output: iteration log, verification results
 *   -> RPS_ConvexTest_database.csv
 *   -> RPS_ConvexTest_viz.html   (open in any browser)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <list>

#include <Eigen/Dense>
#include "gurobi_c++.h"

// ==========================================
// 1. Data Structures
// ==========================================

using Vector = std::vector<double>;

struct SampledCost {
    int id;
    Vector w;
    Vector f;
};

struct Neighborhood {
    std::vector<int> corner_ids;
    double max_regret;
    Vector candidate_w;
    bool is_duplicate;
};

struct RegretResult {
    double max_regret;
    Vector worst_w;
};

struct RegretLogEntry {
    int iteration;
    double max_regret;
    int num_samples;
};

std::ofstream logFile;

void printVector(const std::string& label, const Vector& v) {
    std::cout << label << ": [ ";
    for (auto d : v) std::cout << std::setprecision(6) << d << " ";
    std::cout << "]" << std::endl;
}

void saveDatabaseToCSV(const std::string& filename, const std::vector<SampledCost>& database) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) { std::cerr << "Error: " << filename << std::endl; return; }
    outFile << "ID,W1,W2,W3,F1,F2,F3\n";
    for (const auto& s : database) {
        outFile << s.id;
        for (size_t i = 0; i < s.w.size(); ++i) outFile << "," << s.w[i];
        for (size_t i = 0; i < s.f.size(); ++i) outFile << "," << s.f[i];
        outFile << "\n";
    }
    outFile.close();
    std::cout << "Database saved to: " << filename << std::endl;
}

// ==========================================
// 2. Convex Test Problem (Closed-Form)
// ==========================================

static const double TARGETS[3][2] = {
    { 0.0,  0.0                   },
    {10.0,  0.0                   },
    { 5.0,  5.0 * std::sqrt(3.0) }
};

Vector solveConvexProblem(const Vector& w) {
    double x0 = 0, x1 = 0;
    for (int i = 0; i < 3; ++i) {
        x0 += w[i] * TARGETS[i][0];
        x1 += w[i] * TARGETS[i][1];
    }
    Vector f(3);
    for (int i = 0; i < 3; ++i) {
        double dx = x0 - TARGETS[i][0], dy = x1 - TARGETS[i][1];
        f[i] = dx * dx + dy * dy;
    }
    return f;
}

// ==========================================
// 3. Max Regret LP (Gurobi)
// ==========================================

RegretResult solveMaxRegretLP(const std::vector<SampledCost>& corners,
                              const std::vector<double>& global_max_costs,
                              int num_objectives) {
    int k = corners.size();
    try {
        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "");
        env.start();
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model = GRBModel(env);

        std::vector<GRBVar> lambda(k);
        for (int i = 0; i < k; ++i)
            lambda[i] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "lam");

        std::vector<GRBVar> w(num_objectives);
        for (int j = 0; j < num_objectives; ++j)
            w[j] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "w");

        GRBVar R = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "R");

        for (int j = 0; j < num_objectives; ++j) {
            GRBLinExpr expr = 0;
            for (int i = 0; i < k; ++i) expr += lambda[i] * corners[i].w[j];
            model.addConstr(w[j] == expr);
        }
        GRBLinExpr sumL = 0;
        for (int i = 0; i < k; ++i) sumL += lambda[i];
        model.addConstr(sumL == 1.0);

        std::vector<double> u_corners(k);
        for (int i = 0; i < k; ++i) {
            double dot = 0;
            for (int j = 0; j < num_objectives; ++j)
                dot += corners[i].w[j] * corners[i].f[j] / global_max_costs[j];
            u_corners[i] = dot;
        }
        GRBLinExpr LB = 0;
        for (int i = 0; i < k; ++i) LB += lambda[i] * u_corners[i];

        for (int i = 0; i < k; ++i) {
            GRBLinExpr wfs = 0;
            for (int j = 0; j < num_objectives; ++j)
                wfs += w[j] * corners[i].f[j] / global_max_costs[j];
            model.addConstr(R <= wfs - LB);
        }
        model.setObjective(GRBLinExpr(R), GRB_MAXIMIZE);
        model.optimize();

        Vector rw;
        for (int j = 0; j < num_objectives; ++j)
            rw.push_back(w[j].get(GRB_DoubleAttr_X));
        return {R.get(GRB_DoubleAttr_X), rw};
    } catch (GRBException e) {
        std::cerr << "Gurobi Error: " << e.getMessage() << std::endl;
        return {-1.0, {}};
    }
}

// ==========================================
// 4. Neighborhood Splitting
// ==========================================

bool isLinearlyIndependent(const std::vector<int>& ids,
                           const std::vector<SampledCost>& db, int d) {
    Eigen::MatrixXd A(d, (int)ids.size());
    for (int i = 0; i < (int)ids.size(); ++i)
        for (int j = 0; j < d; ++j)
            A(j, i) = db[ids[i]].w[j];
    return std::abs(A.determinant()) > 1e-9;
}

void splitNeighborhood(const Neighborhood& N, int new_id,
                       const std::vector<SampledCost>& db, int d,
                       const std::vector<double>& gmc,
                       std::list<Neighborhood>& nbs) {
    int n = N.corner_ids.size();
    for (int i = 0; i < n; ++i) {
        Neighborhood ch;
        ch.corner_ids = N.corner_ids;
        ch.corner_ids[i] = new_id;
        if (!isLinearlyIndependent(ch.corner_ids, db, d)) continue;
        std::vector<SampledCost> cs;
        for (int id : ch.corner_ids) cs.push_back(db[id]);
        RegretResult rr = solveMaxRegretLP(cs, gmc, d);
        ch.max_regret = rr.max_regret;
        ch.candidate_w = rr.worst_w;
        ch.is_duplicate = false;
        nbs.push_back(ch);
    }
}

// ==========================================
// 5. Non-Dominance Verification
// ==========================================

bool dominates(const SampledCost& a, const SampledCost& b) {
    bool strict = false;
    for (size_t j = 0; j < a.f.size(); ++j) {
        if (a.f[j] > b.f[j] + 1e-9) return false;
        if (a.f[j] < b.f[j] - 1e-9) strict = true;
    }
    return strict;
}

std::vector<bool> verifyNonDominance(const std::vector<SampledCost>& db) {
    std::vector<bool> nd(db.size(), true);
    int cnt = 0;
    for (size_t i = 0; i < db.size(); ++i) {
        for (size_t j = 0; j < db.size(); ++j) {
            if (i == j) continue;
            if (dominates(db[j], db[i])) { nd[i] = false; cnt++; break; }
        }
    }
    std::cout << "\n--- Non-Dominance Verification ---" << std::endl;
    if (cnt == 0)
        std::cout << "  PASSED: All " << db.size() << " samples are non-dominated." << std::endl;
    else
        std::cout << "  FAILED: " << cnt << " dominated samples found." << std::endl;
    return nd;
}

// ==========================================
// 6. HTML Visualization Generator
// ==========================================

void generateHTML(const std::string& path,
                  const std::vector<SampledCost>& db,
                  const std::vector<bool>& nd,
                  const std::vector<RegretLogEntry>& rlog,
                  const std::vector<double>& gmc) {
    std::ofstream out(path);
    if (!out.is_open()) { std::cerr << "Cannot write " << path << std::endl; return; }

    // --- Build JSON data strings ---
    std::ostringstream jsSamples, jsRegret, jsSurface;

    jsSamples << std::setprecision(10) << "[";
    for (size_t i = 0; i < db.size(); ++i) {
        if (i) jsSamples << ",";
        jsSamples << "{w:[" << db[i].w[0] << "," << db[i].w[1] << "," << db[i].w[2]
                  << "],f:[" << db[i].f[0] << "," << db[i].f[1] << "," << db[i].f[2]
                  << "],nd:" << (nd[i] ? "true" : "false")
                  << ",id:" << db[i].id << "}";
    }
    jsSamples << "]";

    jsRegret << std::setprecision(10) << "[";
    for (size_t i = 0; i < rlog.size(); ++i) {
        if (i) jsRegret << ",";
        jsRegret << "{it:" << rlog[i].iteration << ",r:" << rlog[i].max_regret
                 << ",n:" << rlog[i].num_samples << "}";
    }
    jsRegret << "]";

    int gN = 40;
    jsSurface << std::setprecision(8) << "[";
    bool first = true;
    for (int a = 0; a <= gN; ++a) {
        for (int b = 0; b <= gN - a; ++b) {
            int c = gN - a - b;
            Vector wv = {(double)a/gN, (double)b/gN, (double)c/gN};
            Vector fv = solveConvexProblem(wv);
            if (!first) jsSurface << ",";
            first = false;
            jsSurface << "[" << fv[0] << "," << fv[1] << "," << fv[2] << "]";
        }
    }
    jsSurface << "]";

    // --- Write HTML ---
    out << R"(<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>RPS Convex Verification</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0c0e13;color:#c8cdd8;font-family:'Courier New',monospace;padding:24px}
h1{font-size:15px;color:#4fc4e8;letter-spacing:2px;text-transform:uppercase;
   border-bottom:1px solid #1e2330;padding-bottom:10px;margin-bottom:18px}
h2{font-size:11px;color:#7a8399;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px}
.row{display:flex;gap:14px;margin-bottom:14px;flex-wrap:wrap}
.card{background:#14171e;border:1px solid #1e2330;border-radius:6px;padding:14px;flex:1;min-width:250px}
canvas{display:block;border-radius:4px}
.stat{display:inline-block;padding:3px 10px;border-radius:3px;font-size:11px;font-weight:700;letter-spacing:1px}
.pass{background:#1a2e22;color:#3ac47a;border:1px solid #2a4a34}
.fail{background:#2e1a1a;color:#e8443a;border:1px solid #4a2a2a}
table{width:100%;border-collapse:collapse;font-size:10px;margin-top:8px}
th{text-align:right;padding:4px 6px;color:#4a5166;border-bottom:1px solid #1e2330;font-weight:600}
td{text-align:right;padding:3px 6px;border-bottom:1px solid #111420}
.ndp{color:#3ac47a;font-weight:700;text-align:center}
.ndf{color:#e8443a;font-weight:700;text-align:center}
.legend{font-size:10px;color:#4a5166;margin-bottom:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:5px;vertical-align:middle;margin-right:3px}
#c3d{cursor:grab}
</style></head><body>

<h1>RPS Convex Verification &mdash; f<sub>i</sub>(x) = &Vert;x &minus; a<sub>i</sub>&Vert;&sup2;</h1>

<div class="row" id="statsRow"></div>

<div class="row">
  <div class="card" style="flex:0 0 320px">
    <h2>Weight Simplex Sampling</h2>
    <div class="legend"><i style="background:#e8643a"></i>Corner <i style="background:#4fc4e8;margin-left:10px"></i>RPS sample</div>
    <canvas id="cW" width="310" height="280"></canvas>
  </div>
  <div class="card">
    <h2>3-D Pareto Front (drag to rotate)</h2>
    <div class="legend"><i style="background:rgba(42,90,138,.5)"></i>Analytical surface <i style="background:#4fc4e8;margin-left:10px"></i>RPS samples</div>
    <canvas id="c3d" width="460" height="380"></canvas>
  </div>
</div>

<div class="row">
  <div class="card"><h2>f&#x2081; vs f&#x2082;</h2><canvas id="p01" width="310" height="260"></canvas></div>
  <div class="card"><h2>f&#x2081; vs f&#x2083;</h2><canvas id="p02" width="310" height="260"></canvas></div>
  <div class="card"><h2>f&#x2082; vs f&#x2083;</h2><canvas id="p12" width="310" height="260"></canvas></div>
</div>

<div class="row">
  <div class="card" style="flex:2">
    <h2>Regret Convergence</h2>
    <canvas id="cR" width="600" height="220"></canvas>
  </div>
  <div class="card" style="flex:1;min-width:240px">
    <h2>Non-Dominance Check</h2>
    <div id="ndBox"></div>
  </div>
</div>

<div class="row">
  <div class="card" style="width:100%">
    <h2>Full Database</h2>
    <div style="max-height:380px;overflow-y:auto"><table id="tbl"></table></div>
  </div>
</div>

<script>
)" ;

    // Inject data
    out << "const S=" << jsSamples.str() << ";\n";
    out << "const RL=" << jsRegret.str() << ";\n";
    out << "const SF=" << jsSurface.str() << ";\n";
    out << "const MC=[" << std::setprecision(10)
        << gmc[0] << "," << gmc[1] << "," << gmc[2] << "];\n";

    out << R"(
/* ---------- stats row ---------- */
(function(){
  const nd=S.filter(s=>s.nd).length,dom=S.length-nd;
  const lr=RL[RL.length-1];
  const cards=[
    ['Total Samples', S.length, ''],
    ['Non-Dominated', nd+' / '+S.length, dom===0?'pass':'fail'],
    ['Final Max Regret', lr?lr.r.toExponential(4):'—', '']
  ];
  const el=document.getElementById('statsRow');
  cards.forEach(c=>{
    const d=document.createElement('div');d.className='card';d.style.minWidth='180px';
    d.innerHTML=`<div style="font-size:9px;color:#4a5166;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">${c[0]}</div>`
      +`<div style="font-size:22px;font-weight:700;${c[2]==='pass'?'color:#3ac47a':c[2]==='fail'?'color:#e8443a':'color:#c8cdd8'}">${c[1]}</div>`;
    el.appendChild(d);
  });
})();

/* ---------- weight simplex ---------- */
(function(){
  const cv=document.getElementById('cW'),cx=cv.getContext('2d');
  const W=cv.width,H=cv.height,pad=26,S3=Math.sqrt(3);
  const side=Math.min(W,H)-2*pad;
  const ax=W/2, ay=pad;
  const bx=W/2-side/2, by=pad+side*S3/2;
  const cpx=W/2+side/2, cy=by;
  function toXY(w){return[w[1]*bx+w[2]*cpx+w[0]*ax, w[1]*by+w[2]*cy+w[0]*ay];}

  cx.fillStyle='#0a0c10';
  cx.beginPath();cx.moveTo(ax,ay);cx.lineTo(bx,by);cx.lineTo(cpx,cy);cx.closePath();cx.fill();
  cx.strokeStyle='#1e2330';cx.lineWidth=1;cx.stroke();

  // internal grid
  cx.strokeStyle='#161a24';cx.lineWidth=0.5;
  for(let k=1;k<10;k++){
    const t=k/10;
    const p1=toXY([1-t,t,0]),p2=toXY([1-t,0,t]);cx.beginPath();cx.moveTo(p1[0],p1[1]);cx.lineTo(p2[0],p2[1]);cx.stroke();
    const p3=toXY([0,1-t,t]),p4=toXY([t,1-t,0]);cx.beginPath();cx.moveTo(p3[0],p3[1]);cx.lineTo(p4[0],p4[1]);cx.stroke();
    const p5=toXY([t,0,1-t]),p6=toXY([0,t,1-t]);cx.beginPath();cx.moveTo(p5[0],p5[1]);cx.lineTo(p6[0],p6[1]);cx.stroke();
  }

  S.forEach((s,i)=>{
    const[px,py]=toXY(s.w);
    cx.beginPath();cx.arc(px,py,i<3?6:3.5,0,Math.PI*2);
    cx.fillStyle=i<3?'#e8643a':'#4fc4e8';cx.fill();
    cx.strokeStyle='#0c0e13';cx.lineWidth=0.8;cx.stroke();
  });
  cx.font='bold 11px Courier New';cx.fillStyle='#7a8399';cx.textAlign='center';
  cx.fillText('w\u2081',ax,ay-8);cx.fillText('w\u2082',bx-14,by+14);cx.fillText('w\u2083',cpx+14,cy+14);
})();

/* ---------- 2D projections ---------- */
function draw2D(id,xi,yi,labX,labY){
  const cv=document.getElementById(id),cx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  const p={t:18,r:14,b:36,l:48};
  const pw=W-p.l-p.r,ph=H-p.t-p.b;
  const xM=MC[xi]*1.08,yM=MC[yi]*1.08;
  const sx=v=>p.l+(v/xM)*pw, sy=v=>p.t+ph-(v/yM)*ph;

  cx.fillStyle='#0a0c10';cx.fillRect(p.l,p.t,pw,ph);
  cx.strokeStyle='#1a1f2c';cx.lineWidth=0.5;cx.font='9px Courier New';cx.fillStyle='#4a5166';
  for(let i=0;i<=5;i++){
    let v=i/5*xM;cx.beginPath();cx.moveTo(sx(v),p.t);cx.lineTo(sx(v),p.t+ph);cx.stroke();
    cx.textAlign='center';cx.fillText(Math.round(v),sx(v),p.t+ph+12);
    v=i/5*yM;cx.beginPath();cx.moveTo(p.l,sy(v));cx.lineTo(p.l+pw,sy(v));cx.stroke();
    cx.textAlign='right';cx.fillText(Math.round(v),p.l-4,sy(v)+3);
  }
  SF.forEach(f=>{cx.beginPath();cx.arc(sx(f[xi]),sy(f[yi]),1.4,0,Math.PI*2);cx.fillStyle='rgba(42,90,138,0.35)';cx.fill();});
  S.forEach((s,i)=>{
    cx.beginPath();cx.arc(sx(s.f[xi]),sy(s.f[yi]),i<3?5:3.5,0,Math.PI*2);
    cx.fillStyle=i<3?'#e8643a':'#4fc4e8';cx.fill();
    cx.strokeStyle='#0c0e13';cx.lineWidth=0.8;cx.stroke();
  });
  cx.font='bold 10px Courier New';cx.fillStyle='#7a8399';cx.textAlign='center';
  cx.fillText(labX,p.l+pw/2,H-3);
  cx.save();cx.translate(10,p.t+ph/2);cx.rotate(-Math.PI/2);cx.fillText(labY,0,0);cx.restore();
}
draw2D('p01',0,1,'f\u2081','f\u2082');
draw2D('p02',0,2,'f\u2081','f\u2083');
draw2D('p12',1,2,'f\u2082','f\u2083');

/* ---------- 3D Pareto front ---------- */
(function(){
  const cv=document.getElementById('c3d'),cx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  let aX=-0.45,aY=0.65,drag=false,lx,ly;

  cv.onmousedown=e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing';};
  onmouseup=()=>{drag=false;cv.style.cursor='grab';};
  onmousemove=e=>{
    if(!drag)return;
    aY+=(e.clientX-lx)*0.007;aX+=(e.clientY-ly)*0.007;
    aX=Math.max(-1.3,Math.min(1.3,aX));
    lx=e.clientX;ly=e.clientY;render();
  };

  function proj(f){
    const x=(f[0]/MC[0])*2-1, y=(f[1]/MC[1])*2-1, z=(f[2]/MC[2])*2-1;
    const cY=Math.cos(aY),sY=Math.sin(aY),cX=Math.cos(aX),sX=Math.sin(aX);
    const rx=cY*x+sY*z, ry=y, rz=-sY*x+cY*z;
    const fx=rx, fy=cX*ry-sX*rz, fz=sX*ry+cX*rz;
    const sc=150;
    return{px:W/2+fx*sc, py:H/2-fy*sc, depth:fz};
  }

  function render(){
    cx.clearRect(0,0,W,H);cx.fillStyle='#0a0c10';cx.fillRect(0,0,W,H);

    // axes
    const o=proj([0,0,0]);
    const axes=[[MC[0],0,0,'f\u2081'],[0,MC[1],0,'f\u2082'],[0,0,MC[2],'f\u2083']];
    cx.strokeStyle='#2a3040';cx.lineWidth=1;cx.font='bold 10px Courier New';cx.fillStyle='#7a8399';
    axes.forEach(a=>{
      const e=proj(a);
      cx.beginPath();cx.moveTo(o.px,o.py);cx.lineTo(e.px,e.py);cx.stroke();
      cx.fillText(a[3],e.px+(e.px>o.px?8:-18),e.py-6);
    });

    // surface (depth-sorted)
    const sp=SF.map(f=>({...proj(f),f})).sort((a,b)=>a.depth-b.depth);
    sp.forEach(p=>{
      cx.beginPath();cx.arc(p.px,p.py,1.8,0,Math.PI*2);
      cx.fillStyle='rgba(42,90,138,'+(0.12+0.28*(1+p.depth)/2)+')';cx.fill();
    });

    // RPS samples (depth-sorted)
    const rp=S.map((s,i)=>({...proj(s.f),s,i})).sort((a,b)=>a.depth-b.depth);
    rp.forEach(p=>{
      const r=p.i<3?6:4;
      cx.beginPath();cx.arc(p.px,p.py,r,0,Math.PI*2);
      cx.fillStyle=p.i<3?'#e8643a':'#4fc4e8';cx.fill();
      cx.strokeStyle='rgba(12,14,19,0.7)';cx.lineWidth=1;cx.stroke();
    });
  }
  render();
})();

/* ---------- regret convergence ---------- */
(function(){
  const cv=document.getElementById('cR'),cx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  const p={t:14,r:14,b:36,l:56};
  const pw=W-p.l-p.r,ph=H-p.t-p.b;
  const data=RL.filter(d=>d.r>0);
  if(data.length<2)return;
  const maxR=data[0].r*1.1, maxI=data.length-1;
  const sx=i=>p.l+(i/maxI)*pw, sy=v=>p.t+ph-(v/maxR)*ph;

  cx.fillStyle='#0a0c10';cx.fillRect(p.l,p.t,pw,ph);
  cx.strokeStyle='#1a1f2c';cx.lineWidth=0.5;cx.font='9px Courier New';cx.fillStyle='#4a5166';
  for(let i=0;i<=5;i++){
    const v=i/5*maxR;
    cx.beginPath();cx.moveTo(p.l,sy(v));cx.lineTo(p.l+pw,sy(v));cx.stroke();
    cx.textAlign='right';cx.fillText(v.toFixed(3),p.l-4,sy(v)+3);
  }
  cx.beginPath();cx.strokeStyle='#e8643a';cx.lineWidth=2;
  data.forEach((d,i)=>{i===0?cx.moveTo(sx(i),sy(d.r)):cx.lineTo(sx(i),sy(d.r));});
  cx.stroke();
  data.forEach((d,i)=>{
    cx.beginPath();cx.arc(sx(i),sy(d.r),2.5,0,Math.PI*2);cx.fillStyle='#f09060';cx.fill();
  });
  cx.font='9px Courier New';cx.fillStyle='#4a5166';cx.textAlign='center';
  [0,Math.round(maxI/4),Math.round(maxI/2),Math.round(3*maxI/4),maxI].forEach(i=>{
    if(i<=maxI)cx.fillText(i,sx(i),p.t+ph+13);
  });
  cx.font='bold 10px Courier New';cx.fillStyle='#7a8399';cx.textAlign='center';
  cx.fillText('Iteration',p.l+pw/2,H-4);
  cx.save();cx.translate(10,p.t+ph/2);cx.rotate(-Math.PI/2);cx.fillText('Max Regret',0,0);cx.restore();
})();

/* ---------- non-dominance box ---------- */
(function(){
  const nd=S.filter(s=>s.nd).length,dom=S.length-nd;
  const el=document.getElementById('ndBox');
  el.innerHTML='<div class="stat '+(dom===0?'pass':'fail')+'">'+(dom===0?'ALL PASSED':'FAILED')+'</div>'
    +'<p style="font-size:11px;color:#7a8399;margin-top:10px;line-height:1.6">'
    +nd+' / '+S.length+' samples are non-dominated.<br><br>'
    +(dom===0
      ?'Every weighted-sum solution is verified Pareto-optimal. This confirms correctness of the RPS weight-selection mechanism on a convex problem.'
      :dom+' sample(s) are dominated.')
    +'</p>';
})();

/* ---------- database table ---------- */
(function(){
  const tb=document.getElementById('tbl');
  let h='<thead><tr>';
  ['ID','w\u2081','w\u2082','w\u2083','f\u2081','f\u2082','f\u2083','ND'].forEach(c=>h+='<th>'+c+'</th>');
  h+='</tr></thead><tbody>';
  S.forEach(s=>{
    h+='<tr><td style="color:#4a5166">'+s.id+'</td>';
    s.w.forEach(v=>h+='<td>'+v.toFixed(4)+'</td>');
    s.f.forEach(v=>h+='<td style="color:#4fc4e8">'+v.toFixed(2)+'</td>');
    h+='<td class="'+(s.nd?'ndp':'ndf')+'">'+(s.nd?'\u2713':'\u2717')+'</td></tr>';
  });
  h+='</tbody>';tb.innerHTML=h;
})();
</script></body></html>)";

    out.close();
    std::cout << "Visualization written to: " << path << std::endl;
}

// ==========================================
// 7. Main
// ==========================================

int main() {
    std::string filename = "RPS_ConvexTest_log.txt";
    logFile.open(filename);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return 1;
    }

    std::cout << "============================================" << std::endl;
    std::cout << " RPS Convex Verification Test" << std::endl;
    std::cout << " f_i(x) = ||x - a_i||^2,  i=1,2,3" << std::endl;
    std::cout << " Targets: equilateral triangle, side=10" << std::endl;
    std::cout << "============================================" << std::endl;

    logFile << "Iteration,w1,w2,w3,f1,f2,f3,MaxRegret,is_duplicate\n";

    std::vector<SampledCost> database;
    std::vector<RegretLogEntry> regretLog;
    int num_obj = 3;

    std::vector<Vector> corner_weights = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    std::vector<double> global_max_costs(num_obj, 1.0);

    std::cout << "\n--- Initializing Corners ---" << std::endl;
    for (int i = 0; i < num_obj; ++i) {
        Vector f = solveConvexProblem(corner_weights[i]);
        database.push_back({i, corner_weights[i], f});
        printVector("  Corner w", corner_weights[i]);
        printVector("  Corner f", f);
        for (int k = 0; k < num_obj; ++k)
            if (f[k] > global_max_costs[k]) global_max_costs[k] = f[k];
        logFile << (i - num_obj) << "," << corner_weights[i][0] << "," << corner_weights[i][1]
                << "," << corner_weights[i][2] << "," << f[0] << "," << f[1] << "," << f[2]
                << ",0.0,0\n";
    }

    std::list<Neighborhood> neighborhoods;
    Neighborhood init_n;
    init_n.corner_ids = {0, 1, 2};
    std::vector<SampledCost> init_c;
    for (int id : init_n.corner_ids) init_c.push_back(database[id]);
    RegretResult init_r = solveMaxRegretLP(init_c, global_max_costs, num_obj);
    init_n.max_regret = init_r.max_regret;
    init_n.candidate_w = init_r.worst_w;
    init_n.is_duplicate = false;
    neighborhoods.push_back(init_n);

    std::cout << "\nInitial Max Regret: " << init_n.max_regret << std::endl;
    regretLog.push_back({-1, init_n.max_regret, (int)database.size()});

    int Budget_K = 128;
    double threshold = 0.0001;

    for (int k = 0; k < Budget_K; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;

        double max_r = -1.0;
        auto best = neighborhoods.begin();
        for (auto it = neighborhoods.begin(); it != neighborhoods.end(); ++it) {
            if (it->max_regret > max_r && !it->is_duplicate) {
                max_r = it->max_regret;
                best = it;
            }
        }
        std::cout << "  Max Regret: " << max_r << std::endl;

        if (max_r < threshold) {
            std::cout << "  CONVERGED." << std::endl;
            regretLog.push_back({k, max_r, (int)database.size()});
            break;
        }

        Vector new_w = best->candidate_w;
        Vector new_f = solveConvexProblem(new_w);
        int new_id = database.size();

        printVector("  New w", new_w);
        printVector("  New f", new_f);

        bool dup = false;
        for (size_t i = 0; i < database.size(); ++i) {
            double d = 0;
            for (int j = 0; j < num_obj; ++j) d += std::pow(new_w[j] - database[i].w[j], 2);
            if (std::sqrt(d) < threshold) { dup = true; break; }
        }

        std::vector<int> old_ids = best->corner_ids;

        if (dup) {
            std::cout << "  Duplicate. Discarding." << std::endl;
            neighborhoods.erase(best);
            continue;
        }

        database.push_back({new_id, new_w, new_f});

        logFile << k << "," << new_w[0] << "," << new_w[1] << "," << new_w[2] << ","
                << new_f[0] << "," << new_f[1] << "," << new_f[2] << ","
                << max_r << "," << dup << "\n";
        logFile.flush();

        for (int j = 0; j < num_obj; ++j)
            if (new_f[j] > global_max_costs[j]) global_max_costs[j] = new_f[j];

        neighborhoods.erase(best);
        Neighborhood old_N;
        old_N.corner_ids = old_ids;
        splitNeighborhood(old_N, new_id, database, num_obj, global_max_costs, neighborhoods);

        regretLog.push_back({k, max_r, (int)database.size()});
    }

    // --- Terminal output ---
    std::cout << "\n============================================" << std::endl;
    std::cout << " Final Database (" << database.size() << " samples)" << std::endl;
    std::cout << "============================================" << std::endl;
    for (const auto& s : database) {
        std::cout << "  ID " << std::setw(3) << s.id << "  w=[ ";
        for (auto w : s.w) std::cout << std::setw(8) << std::setprecision(4) << w << " ";
        std::cout << "]  f=[ ";
        for (auto f : s.f) std::cout << std::setw(10) << std::setprecision(4) << f << " ";
        std::cout << "]" << std::endl;
    }

    // --- Non-dominance verification ---
    std::vector<bool> nd = verifyNonDominance(database);
    int num_dom = 0;
    for (bool b : nd) if (!b) num_dom++;

    // --- Consistency check ---
    std::cout << "\n--- Consistency Check ---" << std::endl;
    bool consistent = true;
    for (const auto& s : database) {
        double sum_f = s.f[0] + s.f[1] + s.f[2];
        double x0 = 0, x1 = 0;
        for (int i = 0; i < 3; ++i) {
            x0 += s.w[i] * TARGETS[i][0];
            x1 += s.w[i] * TARGETS[i][1];
        }
        double expected = 3.0*(x0*x0+x1*x1) - 2.0*(x0*15.0+x1*5.0*std::sqrt(3.0)) + 200.0;
        if (std::abs(sum_f - expected) > 1e-9) {
            std::cout << "  WARNING: ID " << s.id << std::endl;
            consistent = false;
        }
    }
    std::cout << (consistent ? "  All PASSED." : "  Some FAILED.") << std::endl;

    // --- Summary ---
    std::cout << "\n============================================" << std::endl;
    std::cout << " Summary" << std::endl;
    std::cout << "  Samples:       " << database.size() << std::endl;
    std::cout << "  Dominated:     " << num_dom << std::endl;
    std::cout << "  Non-Dominance: " << (num_dom == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Consistency:   " << (consistent ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================" << std::endl;

    logFile.close();
    saveDatabaseToCSV("RPS_ConvexTest_database.csv", database);
    generateHTML("RPS_ConvexTest_viz.html", database, nd, regretLog, global_max_costs);

    return 0;
}
