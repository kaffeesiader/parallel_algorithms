#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define N 8

static int cnt=0;

int feasible(int *board, int new_row, int col) {
	// Stelle sicher, dass die neue Dame mit keiner der existierenden
	// Damen auf einer Spalte oder Diagonalen steht.
	for(int i = 0; i < new_row; ++i) {
		if (board[i] == col ||					// Gleiche Spalte
			board[i] + i == col + new_row ||	// Gleiche Diagonale
			board[i] - i == col - new_row)		// Gleiche Diagonale
				return 0;
	}
	return 1;
}

void Nqueens(int *board, int new_row)  //n is the number of the rows
{
	if(new_row==N+1)
		cnt++;
	else
		for(int col=1; col<=N; col++)  //for every column
			if(feasible(board, new_row, col) )
			{
				board[new_row] = col;
				Nqueens(board,new_row+1);  //checking for next row
			}
}

int main()
{
	int board[N];

	Nqueens(board,1);
	printf("%d\n",cnt);
	return 0;
}
