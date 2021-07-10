#include "../book/book.h"

const std::string Book::BOOK_CSS = R"%%(

h1 {
  margin-top:10px;
  margin-bottom:10px;
}

.backLink {
  margin-top:6px;
  margin-bottom:6px;
}

.moveTable {
  display: table;
  border-style: solid;
  border-width: 1px;
  border-collapse: collapse;
}

.moveTableHeader {
  display: table-row;
  font-weight: bold;
  /*
  border-style: solid;
  border-width: 1px;
  border-color: black;
  */
}

.moveTableRow {
  display: table-row;
  /*
  border-style: solid;
  border-width: 1px;
  border-color: black;
  */
}

.moveTableRow:link { text-decoration: none; }
.moveTableRow:visited { text-decoration: none; }
.moveTableRow:hover { text-decoration: none; }
.moveTableRow:active { text-decoration: none; }

.moveTableCell {
  display: table-cell;
  padding: 10px;
  text-decoration:none;
}

.moveTableRow:hover {
  outline-style: solid;
  outline-width: 1px;
  outline-color: black;
}
.moveTableRow.moveHovered {
  outline-style: solid;
  outline-width: 1px;
  outline-color: black;
}

#whoToPlay {
  padding:10px;
}

.stoneShadow {
  display: block;
  opacity: 0.001;
}
.stoneShadow.tableHovered {
  display: block;
  opacity: 0.3;
}
.stoneShadow:hover {
  display: block;
  opacity: 0.3;
}

)%%";

const std::string Book::BOOK_JS = R"%%(

let url = new URL(window.location.href);
let sym = url.searchParams.get("symmetry");
if(!sym)
  sym = 0;

const badnessColors = [
  [0.00, [100,255,245]],
  [0.12, [120,235,130]],
  [0.30, [205,235,60]],
  [0.70, [255,100,0]],
  [1.00, [220,0,0]],
  [2.00, [100,0,0]],
];

function rgba(values,alpha) {
  return "rgba(" + values.join(",") + "," + alpha + ")";
}

function getBadnessColor(bestWinLossValue, winLossDiff, scoreDiff, sqrtPolicyDiff, alpha) {
  let x = (nextPlayer == 1 ? 1 : -1) * (winLossDiff*0.8 + scoreDiff * 0.06) - 0.05 * sqrtPolicyDiff;
  x += Math.max(0.0, (nextPlayer == 1 ? 1 : -1) * 0.5 * bestWinLossValue);

  for(let i = 0; i<badnessColors.length; i++) {
    [x1,c1] = badnessColors[i];
    if(x < x1) {
      if(i <= 0)
        return rgba(c1,alpha);
      [x0,c0] = badnessColors[i-1];
      interp = (x-x0)/(x1-x0);
      return rgba([c0[0] + (c1[0]-c0[0])*interp, c0[1] + (c1[1]-c0[1])*interp, c0[2] + (c1[2]-c0[2])*interp],alpha);
    }
  }
  return rgba(badnessColors[badnessColors.length-1][1],alpha);
}

function getBadnessColorOfMoveIdx(idx, alpha) {
  let moveData = moves[idx];
  let winLossDiff = moveData["winLossValue"] - moves[0]["winLossValue"];
  let scoreDiff = moveData["scoreMean"] - moves[0]["scoreMean"];
  let sqrtPolicyDiff = Math.sqrt(moveData["policy"]) - Math.sqrt(moves[0]["policy"]);
  let moveValueColor = getBadnessColor(moves[0]["winLossValue"], winLossDiff, scoreDiff, sqrtPolicyDiff, alpha);
  return moveValueColor;
}

function getSymPos(pos) {
  let y = Math.floor(pos / boardSizeX);
  let x = pos % boardSizeX;
  if(sym & 1)
    y = boardSizeY-1-y;
  if(sym & 2)
    x = boardSizeX-1-x;
  if(sym >= 4 && boardSizeX == boardSizeY) {
    let tmp = x;
    x = y;
    y = tmp;
  }
  return x + y*boardSizeX;
}
function getInvSymPos(pos) {
  let y = Math.floor(pos / boardSizeX);
  let x = pos % boardSizeX;
  if(sym >= 4 && boardSizeX == boardSizeY) {
    let tmp = x;
    x = y;
    y = tmp;
  }
  if(sym & 1)
    y = boardSizeY-1-y;
  if(sym & 2)
    x = boardSizeX-1-x;
  return x + y*boardSizeX;
}

function compose(sym1,sym2) {
  if(sym1 & 0x4)
    sym2 = (sym2 & 0x4) | ((sym2 & 0x2) >> 1) | ((sym2 & 0x1) << 1);
  return sym1 ^ sym2;
}

function getLinkForPos(pos) {
  let linkPath = links[pos];
  // This is the symmetry we need to add as a GET parameter in the URL for the linked position.
  let symmetryToAlign = linkSymmetries[pos];
  // Except we need to composite it with our current symmetry too.
  symmetryToAlign = compose(symmetryToAlign, sym);
  return linkPath + "?symmetry=" + symmetryToAlign;
}

let body = document.getElementsByTagName("body")[0];
const coordChars = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

let hoverShadowEltsByMove = {};
let hoverTableEltsByMove = {};

{
  let title = document.createElement("h1");
  title.appendChild(document.createTextNode("KataGo Opening Book " + boardSizeX + " x " + boardSizeY + ""));
  title.id = "title";
  body.appendChild(title);
}
{
  let link = document.createElement("div");
  link.classList.add("backLink");
  link.innerHTML = '<a href="/">Back to home page</a> <br/> <a href="../root/root.html">Back to root position</a>';
  body.appendChild(link);
}

let svgNS = "http://www.w3.org/2000/svg";
{
  const pixelsPerTile = 50.0 * Math.sqrt(Math.sqrt((boardSizeX-1)*(boardSizeY-1)) / 8);
  const borderTiles = 0.9;
  const strokeWidth = 0.05;
  const stoneRadius = 0.5;
  const stoneInnerRadius = 0.45;
  const stoneBlackFill = "#222";
  const stoneWhiteFill = "#FFF";
  const markerFontSize = 0.7;
  const coordFontSize = 0.4;
  const coordSpacing = 0.45;
  const backgroundColor = "#DCB35C";

  let boardSvg = document.createElementNS(svgNS, "svg");
  boardSvg.setAttribute("width", pixelsPerTile * (2 * borderTiles + boardSizeX-1));
  boardSvg.setAttribute("height", pixelsPerTile * (2 * borderTiles + boardSizeY-1));
  boardSvg.setAttribute("viewBox", "-" + borderTiles + " -" + borderTiles + " " + (2 * borderTiles + boardSizeX-1) + " " + (2 * borderTiles + boardSizeY-1));

  // Draw background color
  let background = document.createElementNS(svgNS, "rect");
  background.setAttribute("style", "fill:"+backgroundColor);
  background.setAttribute("width", pixelsPerTile * (2 * borderTiles + boardSizeX-1));
  background.setAttribute("height", pixelsPerTile * (2 * borderTiles + boardSizeY-1));
  background.setAttribute("x", -borderTiles);
  background.setAttribute("y", -borderTiles);
  boardSvg.appendChild(background);

  // Draw border of board
  let border = document.createElementNS(svgNS, "rect");
  border.setAttribute("width", boardSizeX-1);
  border.setAttribute("height", boardSizeY-1);
  border.setAttribute("x", 0);
  border.setAttribute("y", 0);
  border.setAttribute("stroke","black");
  border.setAttribute("stroke-width",strokeWidth);
  border.setAttribute("fill","none");
  boardSvg.appendChild(border);

  // Draw internal gridlines of board
  for(let y = 1; y < boardSizeY-1; y++) {
    let stroke = document.createElementNS(svgNS, "path");
    stroke.setAttribute("stroke","black");
    stroke.setAttribute("stroke-width",strokeWidth);
    stroke.setAttribute("fill","none");
    stroke.setAttribute("d","M0,"+y+"h"+(boardSizeX-1));
    boardSvg.appendChild(stroke);
  }
  for(let x = 1; x < boardSizeX-1; x++) {
    let stroke = document.createElementNS(svgNS, "path");
    stroke.setAttribute("stroke","black");
    stroke.setAttribute("stroke-width",strokeWidth);
    stroke.setAttribute("fill","none");
    stroke.setAttribute("d","M"+x+",0v"+(boardSizeY-1));
    boardSvg.appendChild(stroke);
  }

  // Draw star points
  var xStars = [];
  var yStars = [];
  if(boardSizeX < 9)
    xStars = [];
  else if(boardSizeX == 9)
    xStars = [2,boardSizeX-3];
  else if(boardSizeX % 2 == 0)
    xStars = [3,boardSizeX-4];
  else
    xStars = [3,(boardSizeX-1)/2,boardSizeX-4];

  if(boardSizeY < 9)
    yStars = [];
  else if(boardSizeY == 9)
    yStars = [2,boardSizeY-3];
  else if(boardSizeY % 2 == 0)
    yStars = [3,boardSizeY-4];
  else
    yStars = [3,(boardSizeY-1)/2,boardSizeY-4];

  let stars = [];
  for(let y = 0; y<yStars.length; y++) {
    for(let x = 0; x<xStars.length; x++) {
      stars.push([xStars[x],yStars[y]]);
    }
  }
  if(boardSizeY == 9 && boardSizeX == 9) {
    stars.push([4,4]);
  }

  for(const point of stars) {
    let dot = document.createElementNS(svgNS, "path");
    dot.setAttribute("stroke","black");
    dot.setAttribute("stroke-width",0.2);
    dot.setAttribute("stroke-linecap","round");
    dot.setAttribute("d","M"+point[0]+","+point[1]+"l0,0");
    boardSvg.appendChild(dot);
  }

  // Draw stones
  for(let y = 0; y<boardSizeY; y++) {
    for(let x = 0; x<boardSizeX; x++) {
      let pos = y * boardSizeX + x;
      // We need to draw the stones with the given symmetry applied, so for looking up
      // the stone of a position we need to use the inverse symmetry.
      let invSymPos = getInvSymPos(pos);
      if(board[invSymPos] == 1 || board[invSymPos] == 2) {
        // Layer stone color on top of black circle so that we have precise control over the border
        // being a certain radius.
        let stone = document.createElementNS(svgNS, "circle");
        let stoneBorder = document.createElementNS(svgNS, "circle");
        stone.setAttribute("cx",x);
        stone.setAttribute("cy",y);
        stone.setAttribute("r",stoneInnerRadius);
        stoneBorder.setAttribute("cx",x);
        stoneBorder.setAttribute("cy",y);
        stoneBorder.setAttribute("r",stoneRadius);
        stone.setAttribute("stroke","none");
        stoneBorder.setAttribute("stroke","none");
        stoneBorder.setAttribute("fill","black");
        if(board[invSymPos] == 1)
          stone.setAttribute("fill",stoneBlackFill);
        else
          stone.setAttribute("fill",stoneWhiteFill);
        boardSvg.appendChild(stoneBorder);
        boardSvg.appendChild(stone);
      }
    }
  }

  // Draw move labels on board
  for(let i = 0; i<moves.length; i++) {
    let moveData = moves[i];
    if(moveData["move"] == "other" || moveData["move"] == "pass")
      continue;
    for(let j = 0; j<moveData["xy"].length; j++) {
      let xy = moveData["xy"][j];
      let x = xy[0];
      let y = xy[1];
      let pos = y * boardSizeX + x;
      let symPos = getSymPos(pos);
      let symX = symPos % boardSizeX;
      let symY = Math.floor(symPos / boardSizeX);

      // Background-colored circle to mask out the gridlines so that text isn't fighting
      // it for contrast.
      // let lineMask = document.createElementNS(svgNS, "circle");
      // lineMask.setAttribute("cx",symX);
      // lineMask.setAttribute("cy",symY);
      // lineMask.setAttribute("r",stoneRadius);
      // lineMask.setAttribute("stroke","none");
      // lineMask.setAttribute("fill",backgroundColor);
      // boardSvg.appendChild(lineMask);

      // Colored circle based on move value.
      let moveValueGroup = document.createElementNS(svgNS, "g");
      moveValueGroup.setAttribute("opacity",0.85);
      let moveValueColor = getBadnessColorOfMoveIdx(i,1.0);

      let moveValueBigCircle = document.createElementNS(svgNS, "circle");
      moveValueBigCircle.setAttribute("cx",symX);
      moveValueBigCircle.setAttribute("cy",symY);
      moveValueBigCircle.setAttribute("r",0.5*(stoneRadius+stoneRadius));
      moveValueBigCircle.setAttribute("stroke","none");
      moveValueBigCircle.setAttribute("fill",moveValueColor);
      moveValueGroup.appendChild(moveValueBigCircle);

      let moveValueCircleBorder = document.createElementNS(svgNS, "circle");
      moveValueCircleBorder.setAttribute("cx",symX);
      moveValueCircleBorder.setAttribute("cy",symY);
      moveValueCircleBorder.setAttribute("r",0.5*(stoneRadius+stoneRadius));
      moveValueCircleBorder.setAttribute("stroke","none");
      moveValueCircleBorder.setAttribute("fill",nextPlayer == 1 ? "white" : "white");
      moveValueCircleBorder.setAttribute("opacity",0.5);
      moveValueGroup.appendChild(moveValueCircleBorder);

      let moveValueCircle = document.createElementNS(svgNS, "circle");
      moveValueCircle.setAttribute("cx",symX);
      moveValueCircle.setAttribute("cy",symY);
      moveValueCircle.setAttribute("r",stoneInnerRadius);
      moveValueCircle.setAttribute("stroke","none");
      moveValueCircle.setAttribute("opacity",0.95);
      moveValueCircle.setAttribute("fill",moveValueColor);
      moveValueGroup.appendChild(moveValueCircle);

      boardSvg.appendChild(moveValueGroup);

      // Text for marker, centered.
      let marker = document.createElementNS(svgNS, "text");
      marker.textContent = ""+(i+1);
      marker.setAttribute("x",symX);
      marker.setAttribute("y",symY);
      marker.setAttribute("font-size",markerFontSize);
      marker.setAttribute("dominant-baseline","central");
      marker.setAttribute("text-anchor","middle");
      marker.setAttribute("fill",nextPlayer == 1 ? "black" : "white");
      boardSvg.appendChild(marker);

      // Group for hover shadow
      let shadowGroup = document.createElementNS(svgNS, "g");
      shadowGroup.setAttribute("opacity",0.65);
      shadowGroup.setAttribute("moveX",symX);
      shadowGroup.setAttribute("moveY",symY);
      shadowGroup.classList.add("stoneShadow");
      let stoneShadow = document.createElementNS(svgNS, "circle");
      stoneShadow.setAttribute("cx",symX);
      stoneShadow.setAttribute("cy",symY);
      stoneShadow.setAttribute("r",stoneRadius);
      stoneShadow.setAttribute("stroke","none");
      if(nextPlayer == 1)
        stoneShadow.setAttribute("fill",stoneBlackFill);
      else
        stoneShadow.setAttribute("fill",stoneWhiteFill);
      shadowGroup.appendChild(stoneShadow);

      shadowGroup.setAttribute("x",symX);
      shadowGroup.setAttribute("y",symY);
      if(j == 0)
        hoverShadowEltsByMove[i] = shadowGroup;

      shadowGroup.addEventListener("mouseover",(event) => {
        for(let j = 0; j<moves.length; j++) {
          if(j in hoverTableEltsByMove)
            hoverTableEltsByMove[j].classList.remove("moveHovered");
        }
        if(i in hoverTableEltsByMove)
          hoverTableEltsByMove[i].classList.add("moveHovered");
      });
      shadowGroup.addEventListener("mouseout",(event) => {
        if(i in hoverTableEltsByMove)
          hoverTableEltsByMove[i].classList.remove("moveHovered");
      });

      let shadowGroupLink = document.createElementNS(svgNS, "a");
      shadowGroupLink.setAttribute("href",getLinkForPos(pos));
      shadowGroupLink.appendChild(shadowGroup);
      boardSvg.appendChild(shadowGroupLink);
    }
  }

  // Draw board coordinate labels
  for(let y = 0; y < boardSizeY; y++) {
    let label = document.createElementNS(svgNS, "text");
    label.textContent = "" + (boardSizeY-y);
    label.setAttribute("x",-coordSpacing);
    label.setAttribute("y",y);
    label.setAttribute("font-size",coordFontSize);
    label.setAttribute("dominant-baseline","central");
    label.setAttribute("text-anchor","middle");
    label.setAttribute("fill","black");
    boardSvg.appendChild(label);
  }
  for(let y = 0; y < boardSizeY; y++) {
    let label = document.createElementNS(svgNS, "text");
    label.textContent = "" + (boardSizeY-y);
    label.setAttribute("x",boardSizeX-1+coordSpacing);
    label.setAttribute("y",y);
    label.setAttribute("font-size",coordFontSize);
    label.setAttribute("dominant-baseline","central");
    label.setAttribute("text-anchor","middle");
    label.setAttribute("fill","black");
    boardSvg.appendChild(label);
  }
  for(let x = 0; x < boardSizeX; x++) {
    let label = document.createElementNS(svgNS, "text");
    label.textContent = "" + coordChars[x];
    label.setAttribute("x",x);
    label.setAttribute("y",-coordSpacing);
    label.setAttribute("font-size",coordFontSize);
    label.setAttribute("dominant-baseline","central");
    label.setAttribute("text-anchor","middle");
    label.setAttribute("fill","black");
    boardSvg.appendChild(label);
  }
  for(let x = 0; x < boardSizeX; x++) {
    let label = document.createElementNS(svgNS, "text");
    label.textContent = "" + coordChars[x];
    label.setAttribute("x",x);
    label.setAttribute("y",boardSizeY-1+coordSpacing);
    label.setAttribute("font-size",coordFontSize);
    label.setAttribute("dominant-baseline","central");
    label.setAttribute("text-anchor","middle");
    label.setAttribute("fill","black");
    boardSvg.appendChild(label);
  }

  body.appendChild(boardSvg);
}

{
  let whoToPlay = document.createElement("div");
  whoToPlay.appendChild(document.createTextNode((nextPlayer == 1 ? "Black" : "White") + " to play!"));
  whoToPlay.id = "whoToPlay";
  body.appendChild(whoToPlay);
}

function textCell(text) {
  let cell = document.createElement("div");
  cell.classList.add("moveTableCell");
  cell.appendChild(document.createTextNode(text));
  return cell;
}

// Generate move table. Emulate table using divs, so that we can have more precise control via css
// such as linking a whole row.
{
  let table = document.createElement("div");
  table.classList.add("moveTable");

  let headerRow = document.createElement("div");
  headerRow.classList.add("moveTableHeader");
  headerRow.appendChild(textCell("Index"));
  headerRow.appendChild(textCell("Move"));
  headerRow.appendChild(textCell("Black Win%"));
  headerRow.appendChild(textCell("Black Score"));
  headerRow.appendChild(textCell("Lead"));
  headerRow.appendChild(textCell("Win%LCB"));
  headerRow.appendChild(textCell("Win%UCB"));
  headerRow.appendChild(textCell("ScoreLCB"));
  headerRow.appendChild(textCell("ScoreUCB"));
  headerRow.appendChild(textCell("ScoreFinalLCB"));
  headerRow.appendChild(textCell("ScoreFinalUCB"));
  headerRow.appendChild(textCell("Policy%"));
  headerRow.appendChild(textCell("Weight"));
  headerRow.appendChild(textCell("Visits"));
  headerRow.appendChild(textCell("Cost"));
  headerRow.appendChild(textCell("CostFromRoot"));
  table.appendChild(headerRow);

  for(let i = 0; i<moves.length; i++) {
    let moveData = moves[i];
    let dataRow = document.createElement("a");
    dataRow.classList.add("moveTableRow");
    dataRow.setAttribute("role","row");
    dataRow.addEventListener("mouseover",(event) => {
      for(let j = 0; j<moves.length; j++) {
        if(j in hoverShadowEltsByMove)
          hoverShadowEltsByMove[j].classList.remove("tableHovered");
      }
      if(i in hoverShadowEltsByMove)
        hoverShadowEltsByMove[i].classList.add("tableHovered");
    });
    dataRow.addEventListener("mouseout",(event) => {
      if(i in hoverShadowEltsByMove)
        hoverShadowEltsByMove[i].classList.remove("tableHovered");
    });

    if(moveData["xy"]) {
      let xy = moveData["xy"][0];
      let x = xy[0];
      let y = xy[1];
      let pos = y * boardSizeX + x;
      dataRow.setAttribute("href",getLinkForPos(pos));
    }
    else if(moveData["move"] && moveData["move"] == "pass") {
      let pos = boardSizeY * boardSizeX;
      dataRow.setAttribute("href",getLinkForPos(pos));
    }

    dataRow.appendChild(textCell(i+1));
    if(moveData["move"])
      dataRow.appendChild(textCell(moveData["move"]));
    else if(moveData["xy"]) {
      let xy = moveData["xy"][0];
      let x = xy[0];
      let y = xy[1];
      let pos = y * boardSizeX + x;
      let symPos = getSymPos(pos);
      let symX = symPos % boardSizeX;
      let symY = Math.floor(symPos / boardSizeX);
      let coord = coordChars[symX] + "" + (boardSizeY-symY);
      dataRow.appendChild(textCell(coord));
    }
    else
      dataRow.appendChild(textCell(""));

    dataRow.appendChild(textCell((100.0 * (0.5*(1.0-moveData["winLossValue"]))).toFixed(1)+"%"));
    dataRow.appendChild(textCell((-moveData["scoreMean"]).toFixed(2)));
    dataRow.appendChild(textCell((-moveData["lead"]).toFixed(2)));
    dataRow.appendChild(textCell((100.0 * (0.5*(1.0-moveData["winLossUCB"]))).toFixed(1)+"%"));
    dataRow.appendChild(textCell((100.0 * (0.5*(1.0-moveData["winLossLCB"]))).toFixed(1)+"%"));
    dataRow.appendChild(textCell((-moveData["scoreUCB"]).toFixed(2)));
    dataRow.appendChild(textCell((-moveData["scoreLCB"]).toFixed(2)));
    dataRow.appendChild(textCell((-moveData["scoreFinalUCB"]).toFixed(2)));
    dataRow.appendChild(textCell((-moveData["scoreFinalLCB"]).toFixed(2)));
    dataRow.appendChild(textCell((100.0 * moveData["policy"]).toFixed(2)+"%"));
    dataRow.appendChild(textCell(moveData["weight"].toFixed(1)));
    dataRow.appendChild(textCell(moveData["visits"]));
    dataRow.appendChild(textCell(moveData["cost"]));
    dataRow.appendChild(textCell(moveData["costFromRoot"]));

    dataRow.style.background = getBadnessColorOfMoveIdx(i,0.35);

    table.appendChild(dataRow);
    hoverTableEltsByMove[i] = dataRow;
  }

  body.appendChild(table);
}

)%%";
