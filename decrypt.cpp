#include <iostream>
#include <fstream>
#include <string>
#include "bits/stdc++.h"
#include "fptree.hpp"
#include "fptree.cpp"
#define endl '\n'
using namespace std;
int counter=0;
vector<int> process_input(string &s){
    vector<int> result;
    int temp=0;
    int parity=0;
    int flg=0;
    for(int i=0;i<s.size();i++){
        if(s[i]==' '){
            if(parity==1){
                temp*=-1;
            }
            result.push_back(temp);
            counter++;
            temp=0;
            flg=0;
            parity=0;
        }
        else if(s[i]=='-'){
            parity=1;
        }
        else{
            temp*=10;
            temp+=(s[i]-'0');
            flg=1;
        }
    }
    if(flg){
        if(parity==1){
            temp*=-1;
        }
        counter++;
        result.push_back(temp);
    }
    temp=0;
    return result;
}

void create_mappings(map<int,vector<int>> &patterns, ifstream &inputFile){
    string line;
    std::getline(inputFile, line);
    vector<int> temp=process_input(line);
    int mappings=temp[0];
    for(int i=0;i<mappings;i++){
        std::getline(inputFile, line);
        vector<int> temp=process_input(line);
        patterns[temp[0]]=temp;
    }

}

int main(int argc, char *argv[]){
    map<int,vector<int>> mappings;
    std::ifstream inputFile(argv[1]);
    string line;
    create_mappings(mappings,inputFile);
    cerr<<"here";
    freopen(argv[2], "w", stdout);
    while (std::getline(inputFile, line)) {
        // Process each line as needed
        
        vector<int> temp=process_input(line);

        set<int> temp_set(temp.begin(),temp.end());
        while(temp_set.size()>0 && *temp_set.begin()<0){
            int here=*temp_set.begin();
            temp_set.erase(temp_set.find(here));

            for(int i=1;i<mappings[here].size();i++){
                temp_set.insert(mappings[here][i]);
            }

        }
        for(auto &x:temp_set){
            cout<<x<<" ";
        }
        cout<<endl;
    }
    cerr<<counter<<endl;
}