var xb;
var xc;


maximize z: 25*xb + 30*xc;

subject to
hours: xb/200 + xc/140 <= 40;

capB: 0 <= xb <= 6000;
capC: 0 <= xb <= 4000;
