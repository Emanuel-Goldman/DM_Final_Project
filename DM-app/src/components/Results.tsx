import { 
    Alert, 
    BarChart, 
    Container, 
    Header, 
    SpaceBetween, 
    Table, 
    Tabs 
} from "@cloudscape-design/components";
import { RankingResult } from "./types";

interface ResultsProps {
    rankingResults: RankingResult[];
}

export const Results: React.FC<ResultsProps> = ({ rankingResults }) => {
    if (rankingResults.length === 0) {
        return <Alert type="info">Submit your ranking settings to see results.</Alert>;
    }

    // Transform rankingResults into BarChart data format
    const stabilityData = rankingResults.map((ranking, index) => ({
        x: `Ranking ${index + 1}`, // X-axis label (category)
        y: ranking.stability, // Y-axis value (stability score)
    }));

    const weightData = rankingResults.map((ranking, index) => ({
        x: `Ranking ${index + 1}`,
        w1: ranking.ranking_function.w1,
        w2: ranking.ranking_function.w2,
    }));

    return (
        <Container header={<Header variant="h2">Results</Header>}>
            <Tabs
                tabs={[
                    {
                        id: "insights",
                        label: "Insights",
                        content: (
                            <SpaceBetween size="m">
                                {/* Stability Score Bar Chart */}
                                <BarChart
                                    series={[
                                        {
                                            title: "Stability Score",
                                            type: "bar",
                                            data: stabilityData,
                                        },
                                    ]}
                                    xTitle="Rankings"
                                    yTitle="Stability Score"
                                    height={300}
                                    hideFilter
                                    hideLegend
                                />

                                {/* Weight Function Bar Chart */}
                                <BarChart
                                    series={[
                                        {
                                            title: "Weight w1",
                                            type: "bar",
                                            data: weightData.map(d => ({ x: d.x, y: d.w1 })),
                                        },
                                        {
                                            title: "Weight w2",
                                            type: "bar",
                                            data: weightData.map(d => ({ x: d.x, y: d.w2 })),
                                        },
                                    ]}
                                    xTitle="Ranking"
                                    yTitle="Weight Value"
                                    height={300}
                                    hideFilter
                                    hideLegend
                                />
                            </SpaceBetween>
                        ),
                    },
                    {
                        id: "tables",
                        label: "Ranked Tables",
                        content: (
                            <SpaceBetween size="m">
                                {rankingResults.map((result, index) => (
                                    <Table
                                        key={index}
                                        header={
                                            <Header
                                                variant="h3"
                                                description={`Stability: ${result.stability} | Ranking function: w1 = ${result.ranking_function.w1}, w2 = ${result.ranking_function.w2}`}
                                            >
                                                Ranking #{index + 1}
                                            </Header>
                                        }
                                        columnDefinitions={Object.keys(result.ranked_list[0]).map((key) => ({
                                            id: key,
                                            header: key,
                                            cell: (item: any) => item[key],
                                        }))}
                                        items={result.ranked_list}
                                        sortingDisabled
                                        stripedRows
                                        stickyHeader
                                    />
                                ))}
                            </SpaceBetween>
                        ),
                    },
                ]}
            />
        </Container>
    );
};
