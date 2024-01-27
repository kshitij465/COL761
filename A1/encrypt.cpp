#include <iostream>
#include <fstream>
#include <string>
#include "bits/stdc++.h"
#include "fptree.hpp"
#include "fptree.cpp"
#include <chrono>
#define endl '\n'
using namespace std;
int number_of_items=0;
int max_transaction_id=0;
uint64_t minimum_support_threshold;
int pattern_id=-1;
long double time_limit;
auto start_timer = chrono::high_resolution_clock::now();
set<int> process_transaction(const string &s){
    set<int> storage;
    int temp=0;
    int flg=0;
    for(int i=0;i<s.size();i++){
        if(s[i]==' '){
            storage.insert(temp);
            max_transaction_id=max(max_transaction_id,temp);
            temp=0;
            number_of_items+=1;
            flg=0;
        }
        else{
            temp*=10;
            temp+=(s[i]-'0');
            flg=1;
        }
    }
    if(flg){
        storage.insert(temp);
        max_transaction_id=max(max_transaction_id,temp);
    }
    temp=0;
    number_of_items+=1;
    return storage;
}
inline bool is_subset(const set<int> &pattern,set<int> &transaction){
    return includes(transaction.begin(),transaction.end(),pattern.begin(),pattern.end());
    // for(auto &x:pattern){
    //     if(transaction.find(x)==transaction.end()){
    //         return false;
    //     }
    // }
    // return true;
}

void single_step(vector<set<int>> &transactions,map<int,set<int>> &mappings){
    vector<vector<int>> temp_transactions;
    for(auto &x:transactions){
        vector<int> arr(x.begin(),x.end());
        temp_transactions.push_back(arr);
    }
    const FPTree fptree{ temp_transactions, minimum_support_threshold };
    const std::vector<Pattern> patterns = fptree_growth( fptree ,time_limit,start_timer);
    auto current_time=chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(current_time - start_timer);
    long double seconds=((long double)duration.count())/((long double) 1e9);
    // if(seconds>time_limit){
    //     cerr<<seconds;
    //     return;
    // }
    int cntr=0;
    for(auto &x:patterns){
        int freq=x.second;
        int counter=0;
        int flg=0;
        cntr+=1;
        if(cntr==2001){
            break;
        }
        if(x.first.size()>1){
            // cerr<<"f";
            for(auto &y:transactions){
                if(counter==freq){
                    // cerr<<"dalla";
                    break;
                }
                if(is_subset(x.first,y)){
                    counter++;
                    // cerr<<"here"<<endl;
                    for(auto &a:x.first){
                        y.erase(y.find(a));
                    }
                    if(flg==0){
                        y.insert(pattern_id);
                        mappings[pattern_id]=x.first;
                        pattern_id--;
                        flg++;
                    }
                    else{
                        y.insert(pattern_id+1);
                    }
                }

            }
        }
    }
}







int main(int argc, char *argv[]){
    // if (argc != 3) {
    //     cerr<<"Unexpected inputs given.";
    //     return 1;
    // }
    start_timer = chrono::high_resolution_clock::now();
    // Open the file using the filename from the command line argument
    std::ifstream inputFile(argv[1]);
    // std::ifstream inputFile("Transaction_dataset/D_large.dat");
    // if (!inputFile.is_open()) {
    //     std::cerr << "Unable to open file: " << argv[1] << std::endl;
    //     return 1;
    // }

    string line;
    vector<set<int>> transactions;
    // Read lines until there are no lines left
    set<string> temp_trans;
    while (std::getline(inputFile, line)) {
        // Process each line as needed
        // transactions.push_back(process_transaction(line));
        string lm=line;
        temp_trans.insert(lm);
    }
    for(auto &s:temp_trans){
        transactions.push_back(process_transaction(s));
    }
    cerr<<"input_done";
    // for(auto x:transactions){
    //     for (auto a :x){
    //         cerr<<a<<" ";
    //     }
    //     cerr<<endl;
    // }

    // Close the file
    inputFile.close();
    map<int,set<int>> mappings;
    minimum_support_threshold=0.7*transactions.size();
    int temp_minimum_support_threshold=minimum_support_threshold;
    if(number_of_items<2000000){
        time_limit=30;
    }
    else if(number_of_items<23000000){
        time_limit=418;
    }
    else{
        time_limit=1400;
    }
    for(int i=0;;i++){
        minimum_support_threshold=temp_minimum_support_threshold/((i+1)*(i+1));
        single_step(transactions,mappings);
        cerr<<"done"<<endl;
        auto current_time=chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(current_time - start_timer);
        long double seconds=((long double)duration.count())/((long double) 1e9);
        if(seconds>time_limit){
            cerr<<seconds;
            break;
        }

    }
    cerr<<"where"<<endl;
    freopen(argv[2], "w", stdout);

    cout<<-1*pattern_id-1<<endl;
    for(int i=pattern_id+1;i<0;i++){
        cout<<i;
        for(auto &x:mappings[i]){
            cout<<" "<<x;
        }
        cout<<endl;
    }
    for(auto &a:transactions){
        int flg=0;
        for(auto &x:a){
            if(flg==0){
                cout<<x;
                flg=1;
            }
            else{
                cout<<" "<<x;
            }
        }
        cout<<endl;
    }
    max_transaction_id=0;
    number_of_items=0;
    return 0;
}