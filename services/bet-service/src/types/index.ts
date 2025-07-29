export interface Bet {
    id: string;
    userId: string;
    oddsId: string;
    amount: number;
    status: 'pending' | 'won' | 'lost';
    createdAt: Date;
    updatedAt: Date;
}

export interface BetResponse {
    bet: Bet;
    message: string;
}

export interface BetHistory {
    userId: string;
    bets: Bet[];
}