#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <tee_internal_api.h>
#include "option_list.h"
#include "utils.h"

int atoi(char* str); 
int atoi(char* str) 
{ 
    int res = 0; // Initialize result 
  
    // Iterate through all characters of input string and 
    // update result 
    for (int i = 0; str[i] != '\0'; ++i) 
        res = res * 10 + str[i] - '0'; 
  
    // return result. 
    return res; 
} 

float atof(char *arr);
float atof(char *arr){
    int i,j,flag;
    float val;
    char c;
    i=0;
    j=0;
    val=0;
    flag=0;
    while ((c = *(arr+i))!='\0'){
//      if ((c<'0')||(c>'9')) return 0;
        if (c!='.'){
            val =(val*10)+(c-'0');
            if (flag == 1){
                --j;
            }
        }
        if (c=='.'){ if (flag == 1) return 0; flag=1;}
        ++i;
    }
    val = val*ta_pow(10,j);
    return val;
}

#if 0
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    //fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    DMSG("Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}
#endif

int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            //fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
            DMSG("Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

char *option_find(list *l, const char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
char *option_find_str(list *l, const char *key, const char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) 
       // fprintf(stderr, "%s: Using default '%s'\n", key, def);
       DMSG("%s: Using default '%s'\n", key, def);
    return (char*)def;
}

int option_find_int(list *l, const char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
        //fprintf(stderr, "%s: Using default '%d'\n", key, def);
        DMSG("%s: Using default '%d'\n", key, def);
    return def;
}

int option_find_int_quiet(list *l, const char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list *l, const char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list *l, const char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
        //fprintf(stderr, "%s: Using default '%lf'\n", key, def);
        DMSG("%s: Using default '%lf'\n", key, def);
    return def;
}
