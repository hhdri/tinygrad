var currentUOps = 0
async function main() {
  document.addEventListener("keydown", async function(event) {
    if (event.key == 'ArrowLeft') {
      currentUOps = Math.max(0, currentUOps-1)
      main()
    }
    if (event.key == 'ArrowRight') {
      currentUOps += 1
      main()
    }
  })

  const ret = await (await fetch("/"+currentUOps)).json()
  //console.log(ret)
  const g = new dagreD3.graphlib.Graph().setGraph({}).setDefaultEdgeLabel(function() { return {}; });

  for ([k,v] of Object.entries(ret)) {
    //g.setNode(k, {label: v[0]+"\n"+v[3], style: "fill: "+v[4]})
    g.setNode(k, {label: v[0], style: "fill: "+v[4]})
    for (parent of v[2]) {
      g.setEdge(parent, k)
    }
  }

  const render = new dagreD3.render();
  render(d3.select("svg g"), g);
}

main()
