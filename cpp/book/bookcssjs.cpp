#include "../book/book.h"

const std::string Book::BOOK_CSS = R"%%(
.moveTable {
  display: table;
}

.moveTableHeader {
  display: table-row;
}

.moveTableRow {
  display: table-row;
}

.moveTableCell {
  display: table-cell;
  padding: 10px;
}

.moveTableRow:hover {
  background-color: #cccccc;
}

.stoneShadow {
  display: none;
}

)%%";

const std::string Book::BOOK_JS = R"%%(

let url = new URL(window.location.href);
let sym = url.searchParams.get("symmetry");
if(!sym)
  sym = 0;

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
    for(const xy of moveData["xy"]) {
      let x = xy[0];
      let y = xy[1];
      let pos = y * boardSizeX + x;
      let symPos = getSymPos(pos);
      let symX = symPos % boardSizeX;
      let symY = Math.floor(symPos / boardSizeX);

      // Background-colored circle to mask out the gridlines so that text isn't fighting
      // it for contrast.
      let lineMask = document.createElementNS(svgNS, "circle");
      lineMask.setAttribute("cx",symX);
      lineMask.setAttribute("cy",symY);
      lineMask.setAttribute("r",stoneRadius);
      lineMask.setAttribute("stroke","none");
      lineMask.setAttribute("fill",backgroundColor);
      boardSvg.appendChild(lineMask);

      // Group layered stones so that they share the opacity compositely
      let shadowGroup = document.createElementNS(svgNS, "g");
      shadowGroup.setAttribute("opacity",0.65);
      shadowGroup.setAttribute("moveX",symX);
      shadowGroup.setAttribute("moveY",symY);
      shadowGroup.classList.add("stoneShadow");
      let stoneShadow = document.createElementNS(svgNS, "circle");
      let stoneShadowBorder = document.createElementNS(svgNS, "circle");
      stoneShadow.setAttribute("cx",symX);
      stoneShadow.setAttribute("cy",symY);
      stoneShadow.setAttribute("r",stoneInnerRadius);
      stoneShadowBorder.setAttribute("cx",symX);
      stoneShadowBorder.setAttribute("cy",symY);
      stoneShadowBorder.setAttribute("r",stoneRadius);
      stoneShadow.setAttribute("stroke","none");
      stoneShadowBorder.setAttribute("stroke","none");
      stoneShadowBorder.setAttribute("fill","black");
      if(nextPlayer == 1)
        stoneShadow.setAttribute("fill",stoneBlackFill);
      else
        stoneShadow.setAttribute("fill",stoneWhiteFill);
      shadowGroup.appendChild(stoneShadowBorder);
      shadowGroup.appendChild(stoneShadow);
      boardSvg.appendChild(shadowGroup);

      // Create the clickable marker link
      let markerLink = document.createElementNS(svgNS, "a");
      markerLink.setAttribute("href",getLinkForPos(pos));

      // Text inside marker link, centered.
      let marker = document.createElementNS(svgNS, "text");
      marker.textContent = ""+(i+1);
      marker.setAttribute("x",symX);
      marker.setAttribute("y",symY);
      marker.setAttribute("font-size",markerFontSize);
      marker.setAttribute("dominant-baseline","central");
      marker.setAttribute("text-anchor","middle");
      if(nextPlayer == 1)
        marker.setAttribute("fill","black");
      else
        marker.setAttribute("fill","white");

      markerLink.appendChild(marker);
      boardSvg.appendChild(markerLink);
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
  headerRow.appendChild(textCell("Win%"));
  headerRow.appendChild(textCell("Score"));
  headerRow.appendChild(textCell("Lead"));
  headerRow.appendChild(textCell("Win%LCB"));
  headerRow.appendChild(textCell("Win%UCB"));
  headerRow.appendChild(textCell("ScoreLCB"));
  headerRow.appendChild(textCell("ScoreUCB"));
  headerRow.appendChild(textCell("Policy%"));
  headerRow.appendChild(textCell("Weight"));
  headerRow.appendChild(textCell("Visits"));
  headerRow.appendChild(textCell("Cost"));
  table.appendChild(headerRow);

  for(let i = 0; i<moves.length; i++) {
    let moveData = moves[i];
    let dataRow = document.createElement("a");
    dataRow.classList.add("moveTableRow");
    dataRow.setAttribute("role","row");

    if(moveData["xy"]) {
      let xy = moveData["xy"][0];
      let x = xy[0];
      let y = xy[1];
      let pos = y * boardSizeX + x;
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
    dataRow.appendChild(textCell((100.0 * moveData["policy"]).toFixed(2)+"%"));
    dataRow.appendChild(textCell(moveData["weight"].toFixed(1)));
    dataRow.appendChild(textCell(moveData["visits"]));
    dataRow.appendChild(textCell(moveData["cost"]));
    table.appendChild(dataRow);
  }

  body.appendChild(table);
}


)%%";
