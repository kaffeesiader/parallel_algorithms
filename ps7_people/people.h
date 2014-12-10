#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAX_AGE 120
#define NAME_LEN 32

typedef char name_t[NAME_LEN];

typedef struct {
	int age;
	name_t name;
} person_t;

// Name Generation ------------------------------------------------------------

#define BUF_SIZE 16
#define FIRST_NAMES_FILE "first_names.txt"
#define LAST_NAMES_FILE "last_names.txt"

int count_lines(const char *filename) {
	int lines = 0;
	FILE *f = fopen(filename, "r");
	assert(f != NULL);
	char c;
	while((c = fgetc(f)) != EOF) if(c == '\n') lines++;
	fclose(f);
	if(c != '\n') lines++;
	return lines;
}

int load_names(const char *filename, char ***storage) {
	int lines = count_lines(filename) - 1;
	*storage = (char**)malloc(lines * sizeof(char*));
	char *space =  (char*)malloc(lines * BUF_SIZE * sizeof(char));
	
	FILE *f = fopen(filename, "r");
	for(int i=0; i<lines; ++i) {
		(*storage)[i] = space + i * BUF_SIZE;
		assert(fgets((*storage)[i], BUF_SIZE-1, f) != NULL);
		// remove whitespace chars, if any
		char *c;
		while((c = strchr((*storage)[i], '\n'))) *c = '\0';
		while((c = strchr((*storage)[i], '\r'))) *c = '\0';
		while((c = strchr((*storage)[i], ' '))) *c = '\0';
	}
	fclose(f);
	return lines;
}

void gen_name(name_t buffer) {
	static char** first_names = NULL;
	static char** last_names = NULL;
	static int first_name_count, last_name_count;
	if(first_names == NULL) { // initialize on first call
		first_name_count = load_names(FIRST_NAMES_FILE, &first_names);
		last_name_count = load_names(LAST_NAMES_FILE, &last_names);
	}

	snprintf(buffer, NAME_LEN, "%s %s",
		first_names[rand()%first_name_count], 
		last_names[rand()%last_name_count]);
}

void gen_person(person_t *person) {
	gen_name(person->name);
	person->age = rand()%MAX_AGE;
}

void generate_list(person_t **list, int entries) {
	for (int i = 0; i < entries; ++i) {
		list[i] = malloc(sizeof(person_t));
		gen_person(list[i]);
	}
}

void fill_list(person_t *list, int n) {
	for(int i = 0; i < n; ++i)
		gen_person(&list[i]);
}

void free_list(person_t **list, int entries) {
	for (int i = 0; i < entries; ++i) {
		free(list[i]);
	}
}

void print_list(person_t **list, int entries) {
	printf("\n");
	printf("Age | %-32s |\n", "Name");
	printf("----|----------------------------------|\n");
	for (unsigned int i = 0; i < entries; ++i) {
		printf("%3d | %-32s |\n", list[i]->age, list[i]->name);
	}
	printf("\n");
}

void print_persons(person_t *list, int entries) {
	printf("\n");
	printf("Age | %-32s |\n", "Name");
	printf("----|----------------------------------|\n");
	for (unsigned int i = 0; i < entries; ++i) {
		printf("%3d | %-32s |\n", list[i].age, list[i].name);
	}
	printf("\n");
}
