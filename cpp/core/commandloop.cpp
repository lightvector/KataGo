#include "../core/commandloop.h"

//------------------------
#include "../core/using.h"
//------------------------

string CommandLoop::processSingleCommandLine(const string& s) {
  string line = Global::trim(s);

  //Filter down to only "normal" ascii characters. Also excludes carrage returns and newlines
  size_t newLen = 0;
  for(size_t i = 0; i < line.length(); i++)
    if(((int)line[i] >= 32 && (int)line[i] <= 126) || line[i] == '\t')
      line[newLen++] = line[i];

  line.erase(line.begin()+newLen, line.end());

  //Remove comments
  size_t commentPos = line.find("#");
  if(commentPos != string::npos)
    line = line.substr(0, commentPos);

  //Convert tabs to spaces
  for(size_t i = 0; i < line.length(); i++)
    if(line[i] == '\t')
      line[i] = ' ';

  line = Global::trim(line);
  return line;
}
