export type Odds = {
    id: string;
    event: string;
    oddsValue: number;
    bookmaker: string;
};

export interface OddsResponse {
    success: boolean;
    data: Odds[];
    message?: string;
}