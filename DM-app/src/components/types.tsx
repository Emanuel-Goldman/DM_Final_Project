import { SelectProps } from "@cloudscape-design/components";

export type Option = SelectProps.Option;

export interface RankingResult {
    ranked_list: RankedItem[];
    ranking_function: RankingFunction;
    stability: number;
}

export interface RankingFunction {
    w1: number;
    w2: number;
}

export interface RankedItem {
    user_id: number;
    name: string;
    rank: number;
    [key: string]: any;
}