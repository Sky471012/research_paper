#include<iostream>
#include<map>
#include<vector>
#include<algorithm>
#include<sstream>
#include<string>
using namespace std;

map<char, vector<string>> grammar;
int pos = 0;
string input;

int main() {

    int n;
    cout<<"Enter no. of productios:- ";
    cin>>n;

    cin.ignore();

    cout<<"Enter :- "<<endl;
    for(int i=0; i<n; i++){
        string line;
        getline(cin, line);

        char lhs = line[0];
        string rhs = line.substr(5);

        stringstream ss(rhs);
        string prod;

        while(getline(ss, prod, '|')) {
            prod.erase(remove(prod.begin(), prod.end(), ' '), prod.end());
            grammar[lhs].push_back(prod);
        }
    }

    return 0;
}