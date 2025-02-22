import { Alert, Button, Container, FormField, Header, Input, SpaceBetween, Textarea } from "@cloudscape-design/components";
import axios from "axios";
import { useEffect, useState } from "react";
import { RESOURCES, PORT, API_ENDPOINT } from "../consts/apiConsts";
import { Option } from "./types";
import { AttributeSelect, useAttributeSelection } from "./AttributeSelect";
import { AlgorithmSelect, useAlgorithmSelection } from "./AlgorithmSelect";
import { useConstraintInput } from "./ConstraintInput";
import { Results } from "./Results";

export const RankingSettingsSection: React.FC = () => {
    const [columns, setColumns] = useState<Option[]>([]);
    const {
        firstAttr,
        handleFirstAttrSelect,
        secondAttr,
        handleSecondAttrSelect
    } = useAttributeSelection();
    const {
        algorithm,
        handleAlgorithmSelect
    } = useAlgorithmSelection();
    const {
        constraints,
        parsedConstraints,
        constraintError,
        parseConstraints,
        handleConstraintsChange
    } = useConstraintInput();
    const [tupleLimit, setTupleLimit] = useState<string>('5');
    const [rankingCount, setRankingCount] = useState<string>('10');
    
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [rankingResults, setRankingResults] = useState<any[]>([]);

    // Fetches dataset columns on initial render
    useEffect(() => {
        const fetchColumns = async () => {
            setError(null);
            try {
                const res = await axios.get(`${API_ENDPOINT}:${PORT}/${RESOURCES.COLUMNS}`);
                
                // Map columns to { label, value } format
                const columnOptions: Option[] = JSON.parse(res.data).map((col: string) => ({
                    label: col,
                    value: col,
                }));

                setColumns(columnOptions);
            } catch (err: any) {
                setError("Error fetching columns");
                console.error("API Error:", err);
            } 
        };

        if (columns.length == 0) {
            fetchColumns();
        }
    }, []);

    const handleSubmit = async () => {
        parseConstraints();

        setError(null);
        setLoading(true);

        try {
            const data = {
                constraints: parsedConstraints,
                method: algorithm?.value,
                columns: [firstAttr?.value, secondAttr?.value],
                num_ret_tuples: parseInt(tupleLimit, 10),
                num_of_rankings: parseInt(rankingCount, 10)
            };

            const res = await axios.post(`${API_ENDPOINT}:${PORT}/${RESOURCES.RANKING}`, data);
            console.log("res", res); // DEBUG
            setRankingResults(res.data);
        } catch (err: any) {
            setError("Error fetching data");
            console.error("API Error:", err);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <SpaceBetween size="m">
            <Container
                header={
                    <Header 
                        variant="h2"
                        description="Select two attributes, a ranking algorithm, and supply constraints."
                    >
                        Ranking settings
                    </Header>
                }
            >
                <SpaceBetween size="s">
                    <SpaceBetween size="s" direction="horizontal">
                        <AttributeSelect
                            label="Attribute 1"
                            description="First scoring attribute"
                            selectedAttr={firstAttr}
                            columns={columns}
                            handleSelection={handleFirstAttrSelect}
                        />
                        <AttributeSelect
                            label="Attribute 2"
                            description="Second scoring attribute"
                            selectedAttr={secondAttr}
                            columns={columns}
                            handleSelection={handleSecondAttrSelect}
                        />
                        <AlgorithmSelect
                            label="Ranking algorithm"
                            selectedAlg={algorithm}
                            handleSelection={handleAlgorithmSelect}
                        />
                        <FormField 
                            label="Ranking count"
                            description="Number of most stable rankings to return"
                        >
                            <Input
                                onChange={({ detail }) => setRankingCount(detail.value)}
                                value={rankingCount}
                                inputMode="numeric"
                                type="number"
                            />
                        </FormField>
                        <FormField 
                            label="Tuple limit"
                            description="Number of tuples to return in each ranking"
                        >
                            <Input
                                onChange={({ detail }) => setTupleLimit(detail.value)}
                                value={tupleLimit}
                                inputMode="numeric"
                                type="number"
                            />
                        </FormField>
                    </SpaceBetween>
                    <FormField 
                        label={"Constraints"}
                        description={`Expected format: "number*w1 operator number*w2" (without spaces). Example: 1*w1>=2*w2.`}
                        constraintText={`Every constraint should appear in a separate line.`}
                    >
                        {error && <Alert type="error">{constraintError}</Alert>}
                        <Textarea
                            placeholder="Enter constraints for the weight function..."
                            value={constraints}
                            onChange={handleConstraintsChange}
                            invalid={constraintError != null}
                        />
                    </FormField>
                    <Button
                        disabled={!firstAttr || !secondAttr || !algorithm}
                        disabledReason="You must select two attributes and an algorithm to submit"
                        onClick={handleSubmit}
                    >
                        Submit
                    </Button>
                </SpaceBetween>
            </Container>
            <Results rankingResults={rankingResults}/>
        </SpaceBetween>
    );
}