#include "../core/base64.h"

#include <cctype>
#include "../core/test.h"

using namespace std;

static const char* base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static constexpr int decodeTable[128] = {
  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,62,-1,-1,-1,63,
  52,53,54,55,56,57,58,59, 60,61,-1,-1,-1,-1,-1,-1,
  -1, 0, 1, 2, 3, 4, 5, 6,  7, 8, 9,10,11,12,13,14,
  15,16,17,18,19,20,21,22, 23,24,25,-1,-1,-1,-1,-1,
  -1,26,27,28,29,30,31,32, 33,34,35,36,37,38,39,40,
  41,42,43,44,45,46,47,48, 49,50,51,-1,-1,-1,-1,-1
};

string Base64::encode(const string& s) {
  string result;
  size_t size = s.size();
  result.reserve(((size+2)/3)*4);
  for(size_t i = 0; i<size; i += 3) {
    int c0 = (int)s[i] & 0xff;
    int c1 = (int)((i+1 < size) ? s[i+1] : '\0') & 0xff;
    int c2 = (int)((i+2 < size) ? s[i+2] : '\0') & 0xff;

    int e0 = c0 >> 2;
    int e1 = ((c0 & 0x3) << 4) | (c1 >> 4);
    int e2 = ((c1 & 0xf) << 2) | (c2 >> 6);
    int e3 = c2 & 0x3f;

    result += base64Chars[e0];
    result += base64Chars[e1];
    result += base64Chars[e2];
    result += base64Chars[e3];
  }
  size_t excess = size - (size / 3 * 3);
  if(excess == 2) {
    result[result.size()-1] = '=';
  }
  else if(excess == 1) {
    result[result.size()-1] = '=';
    result[result.size()-2] = '=';
  }
  return result;
}

string Base64::decode(const string& s) {
  string result;
  size_t size = s.size();
  result.reserve(((size+3)/4)*3);

  int carryNumBits = 0;
  int carry = 0;
  size_t i;
  for(i = 0; i<size; i++) {
    char c = s[i];
    if(c == '=')
      break;
    if(c < '+' || c > 'z')
      throw StringError(string("Base64::decode: invalid character ") + c);
    int d = decodeTable[(int)c];
    if(d < 0)
      throw StringError(string("Base64::decode: invalid character ") + c);
    // cout << "Got: " << c << " " << d << endl;

    carryNumBits += 6;
    carry = (carry << 6) | d;
    if(carryNumBits >= 8) {
      int extracted = carry >> (carryNumBits-8);
      // cout << "Extracted: " << extracted << endl;
      result += (char)extracted;
      carry = carry ^ (extracted << (carryNumBits-8));
      carryNumBits -= 8;
    }
  }
  for(; i<size; i++) {
    char c = s[i];
    if(c != '=') {
      throw StringError("Base64::decode: string contains other characters after '='");
    }
  }
  if(carry != 0)
    throw StringError("Base64::decode: unexpected end of decode, carry is nonzero");

  return result;
}


void Base64::runTests() {
  cout << "Running base64 tests" << endl;
  {
    const char* name = "base64 tests";
    ostringstream out;

    auto safePrint = [&](const string& s) {
      for(size_t i = 0; i<s.size(); i++) {
        if(std::isprint(s[i]))
          out << s[i];
        else
          out << "(" << (int)s[i] << ")";
      }
    };
    auto runTest = [&](const string& s) {
      string encoded = encode(s);
      string decoded = decode(encoded);
      testAssert(decoded == s);
      safePrint(s);
      out << " : " << encoded << endl;
    };
    auto runDecode = [&](const string& s) {
      try {
        string decoded = decode(s);
        string encoded = encode(decoded);
        string decoded2 = decode(encoded);
        testAssert(decoded == decoded2);
        safePrint(s);
        out << " -> ";
        safePrint(decoded);
        out << endl;
      }
      catch(const StringError& e) {
        safePrint(s);
        out << " error: " << e.what() << endl;
      }
    };

    runTest("");
    runTest("pleasure.");
    runTest("leasure.");
    runTest("easure.");
    runTest("asure.");
    runTest("sure.");
    runTest("ure.");
    runTest("re.");
    runTest("e.");
    runTest(".");
    runTest("\xFF");
    runTest("\xFF\x01");
    runTest("\xFF\x01\xFF");
    runTest("\xFF\x01\xFF\xFF");
    runTest("\xFF\x01\xFF\xFF\x01");
    runTest("\xFF\x01\xFF\xFF\x01\xFF");
    runTest("\xFF\x01\xFF\xFF\x01\xFF\xFF");
    runTest("\xFF\x01\xFF\xFF\x01\xFF\xFF\xFF");
    runTest("\xFF\x01\xFF\xFF\x01\xFF\xFF\xFF\x01");

    string tmp;
    tmp += (char)(0);
    tmp += (char)(100);
    tmp += (char)(0);
    tmp += (char)(100);
    runTest(tmp);

    runDecode("YWJjZGVm");
    runDecode("YWJjZGVm=");
    runDecode("YWJjZGVm==");
    runDecode("YWJjZGVm===");
    runDecode("YWJjZGU");
    runDecode("YWJjZGU=");
    runDecode("YWJjZGU==");
    runDecode("YWJjZGU===");
    runDecode("YWJjZA");
    runDecode("YWJjZA=");
    runDecode("YWJjZA==");
    runDecode("YWJjZA===");

    runTest("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.");

    runDecode("YWJjZB=");
    runDecode("YWJjZC=");
    runDecode("YWJjZD=");
    runDecode("YWJjZE=");
    runDecode("YWJjZF=");
    runDecode("YWJjZG=");
    runDecode("YWJjZH=");
    runDecode("YWJjZI=");
    runDecode("YWJjZJ=");
    runDecode("YWJjZK=");
    runDecode("YWJjZL=");
    runDecode("YWJjZM=");
    runDecode("YWJjZN=");
    runDecode("YWJjZO=");
    runDecode("YWJjZP=");
    runDecode("YWJjZQ=");
    runDecode("YWJjZR=");

    runDecode("YWJj!");
    runDecode("YWJjZGVm\n");

    string expected = R"%%(
 :
pleasure. : cGxlYXN1cmUu
leasure. : bGVhc3VyZS4=
easure. : ZWFzdXJlLg==
asure. : YXN1cmUu
sure. : c3VyZS4=
ure. : dXJlLg==
re. : cmUu
e. : ZS4=
. : Lg==
(-1) : /w==
(-1)(1) : /wE=
(-1)(1)(-1) : /wH/
(-1)(1)(-1)(-1) : /wH//w==
(-1)(1)(-1)(-1)(1) : /wH//wE=
(-1)(1)(-1)(-1)(1)(-1) : /wH//wH/
(-1)(1)(-1)(-1)(1)(-1)(-1) : /wH//wH//w==
(-1)(1)(-1)(-1)(1)(-1)(-1)(-1) : /wH//wH///8=
(-1)(1)(-1)(-1)(1)(-1)(-1)(-1)(1) : /wH//wH///8B
(0)d(0)d : AGQAZA==
YWJjZGVm -> abcdef
YWJjZGVm= -> abcdef
YWJjZGVm== -> abcdef
YWJjZGVm=== -> abcdef
YWJjZGU -> abcde
YWJjZGU= -> abcde
YWJjZGU== -> abcde
YWJjZGU=== -> abcde
YWJjZA -> abcd
YWJjZA= -> abcd
YWJjZA== -> abcd
YWJjZA=== -> abcd
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. : TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQsIGNvbnNlY3RldHVyIGFkaXBpc2NpbmcgZWxpdCwgc2VkIGRvIGVpdXNtb2QgdGVtcG9yIGluY2lkaWR1bnQgdXQgbGFib3JlIGV0IGRvbG9yZSBtYWduYSBhbGlxdWEuIFV0IGVuaW0gYWQgbWluaW0gdmVuaWFtLCBxdWlzIG5vc3RydWQgZXhlcmNpdGF0aW9uIHVsbGFtY28gbGFib3JpcyBuaXNpIHV0IGFsaXF1aXAgZXggZWEgY29tbW9kbyBjb25zZXF1YXQuIER1aXMgYXV0ZSBpcnVyZSBkb2xvciBpbiByZXByZWhlbmRlcml0IGluIHZvbHVwdGF0ZSB2ZWxpdCBlc3NlIGNpbGx1bSBkb2xvcmUgZXUgZnVnaWF0IG51bGxhIHBhcmlhdHVyLiBFeGNlcHRldXIgc2ludCBvY2NhZWNhdCBjdXBpZGF0YXQgbm9uIHByb2lkZW50LCBzdW50IGluIGN1bHBhIHF1aSBvZmZpY2lhIGRlc2VydW50IG1vbGxpdCBhbmltIGlkIGVzdCBsYWJvcnVtLg==
YWJjZB= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZC= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZD= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZE= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZF= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZG= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZH= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZI= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZJ= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZK= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZL= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZM= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZN= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZO= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZP= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZQ= -> abce
YWJjZR= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJj! error: Base64::decode: invalid character !
YWJjZGVm(10) error: Base64::decode: invalid character
)%%";
    TestCommon::expect(name,out,expected);
  }

}

